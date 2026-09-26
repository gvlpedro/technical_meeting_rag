"""Every endpoint the Streamlit app in `frontend/` calls. This is one router, not five.
All of it exists to serve that one app's four tabs (see the README's "User interface"
section). Splitting it further would only scatter one cohesive story across more files.

Auth (`/login`) checks against `settings.frontend_users`. This is a fixed, hardcoded
list of logins, with no user table (see `FrontendUser`'s own docstring in
`app/config.py`). `tenant` is what actually keeps data separate between logins.
Nothing here issues a session token. Instead, the frontend just holds
`{username, tenant}` in its own `st.session_state` after a successful login, and sends
`tenant` back on every later call.
"""

import asyncio
import json
import re
from datetime import date
from pathlib import Path
from typing import Literal
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.types import Command
from pydantic import BaseModel
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from agents.graph import _persist_document_version, _top_questions, build_graph, checkpointer_dsn
from agents.shared import (
    bronze_content_for_source,
    bronze_ingestion_date_for_source,
    insert_authors_line,
    insert_source_line,
    qa_pairs_for_source,
)
from agents.stages import gold
from agents.stages.adr_critic.prompts import build_critic_prompt
from agents.stages.adr_critic.schemas import CritiqueResult
from agents.stages.adr_generation.prompts import build_adr_generation_prompt
from agents.stages.adr_generation.service import own_previous_architecture_diagram
from agents.stages.architecture_questions.service import (
    generate_architecture_questions_for_batch,
    previous_architecture_context,
)
from agents.stages.classification.prompts import build_classification_prompt
from agents.stages.classification.schemas import ClassificationResult
from agents.stages.data_contract_questions.service import generate_data_contract_questions_for_batch
from agents.state import initial_state
from agents.template import load_json_response
from app.config import settings
from db.models import GoldEvolution, LlmCost, SilverChunk, SilverDocument
from db.session import get_session
from ingestion.embedder import embed
from ingestion.service import NoTranscriptsFoundError, ingest_uploaded_files
from llm.router import bind_tenant
from llm.router import complete as llm_complete

router = APIRouter(prefix="/v1/frontend", tags=["frontend"])


# --- Auth --------------------------------------------------------------------------------


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    username: str
    tenant: str


@router.post("/login", response_model=LoginResponse)
async def login(request: LoginRequest) -> LoginResponse:
    for user in settings.frontend_users:
        if user.username == request.username and user.password == request.password:
            return LoginResponse(username=user.username, tenant=user.tenant)
    raise HTTPException(status_code=401, detail="Invalid username or password")


def _tenant_for_username(username: str) -> str:
    """The one place a `username` turns into the tenant"""
    for user in settings.frontend_users:
        if user.username == username:
            return user.tenant
    raise HTTPException(status_code=401, detail="Unknown user")


class ConfigResponse(BaseModel):
    start_test_mode: bool


@router.get("/config", response_model=ConfigResponse)
async def config() -> ConfigResponse:
    """Lets the frontend decide whether to show the monitor tab at all"""
    return ConfigResponse(start_test_mode=settings.start_test_mode)


# --- Input transcription -------------------------------------------------------------------


class TranscriptionResponse(BaseModel):
    ingestion_date: str
    files_ingested: list[str] = []
    status: Literal["completed", "pending_review"]
    thread_id: str
    pending_questions: list[str] = []
    documents: dict[str, str] = {}
    # This maps source_component to the Critic's completeness_score (0-100)
    scores: dict[str, int] = {}
    unresolved_points: dict[str, list[str]] = {}


class ResumeRequest(BaseModel):
    thread_id: str
    answers: dict[str, str]
    username: str


async def _run_graph(thread_id: str, payload: dict) -> dict:
    async with AsyncPostgresSaver.from_conn_string(checkpointer_dsn()) as saver:
        await saver.setup()
        graph = build_graph(saver)
        return await graph.ainvoke(payload, config={"configurable": {"thread_id": thread_id}})


async def _resume_graph(thread_id: str, answers: dict[str, str], *, expected_tenant: str) -> dict:
    """Checks, via `graph.aget_state`, that this `thread_id`'s saved tenant matches
    `expected_tenant` before resuming it — since `thread_id` is just a guessable string, not a
    secret, this stops a request from resuming and reading back another tenant's paused
    draft."""
    async with AsyncPostgresSaver.from_conn_string(checkpointer_dsn()) as saver:
        await saver.setup()
        graph = build_graph(saver)
        config = {"configurable": {"thread_id": thread_id}}
        snapshot = await graph.aget_state(config)
        if snapshot.values.get("tenant") != expected_tenant:
            raise HTTPException(
                status_code=404, detail="No paused clarification session found for this thread"
            )
        return await graph.ainvoke(Command(resume=answers), config=config)


def _to_response(thread_id: str, ingestion_date: str, files_ingested: list[str], state: dict) -> TranscriptionResponse:
    if "__interrupt__" in state:
        pending = state["__interrupt__"][0].value["pending_questions"]
        return TranscriptionResponse(
            ingestion_date=ingestion_date,
            files_ingested=files_ingested,
            status="pending_review",
            thread_id=thread_id,
            pending_questions=pending,
        )
    return TranscriptionResponse(
        ingestion_date=ingestion_date,
        files_ingested=files_ingested,
        status="completed",
        thread_id=thread_id,
        documents=state.get("documents", {}),
        scores=state.get("adr_scores", {}),
        unresolved_points=state.get("adr_unresolved_points", {}),
    )


@router.post("/transcriptions/upload", response_model=TranscriptionResponse)
async def upload_transcription(
    ingestion_date: str = Form(...),
    max_questions_per_stage: int = Form(...),
    username: str = Form(...),
    files: list[UploadFile] = File(...),
    db: AsyncSession = Depends(get_session),
) -> TranscriptionResponse:
    """Ingests every uploaded file into Bronze (capped at `settings.max_upload_file_bytes`,
    tenant derived from `username`) and runs the Silver clarification loop on it with
    `persist=False` — so the result is only a draft until "Publish" — pausing with
    `status: "pending_review"` (answered via `/resume`) if a human needs to resolve up to
    `max_questions_per_stage` questions per stage."""
    tenant = _tenant_for_username(username)
    payloads: list[tuple[str, bytes]] = []
    for f in files:
        content = await f.read(settings.max_upload_file_bytes + 1)
        if len(content) > settings.max_upload_file_bytes:
            limit_mb = settings.max_upload_file_bytes // (1024 * 1024)
            raise HTTPException(
                status_code=413,
                detail=f"{f.filename or 'upload'!r} exceeds the {limit_mb} MB upload limit",
            )
        payloads.append((f.filename or "upload", content))
    try:
        result = await ingest_uploaded_files(payloads, ingestion_date, tenant, db, uploaded_by=username)
    except (ValueError, NoTranscriptsFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    bind_tenant(tenant)
    thread_id = f"frontend-{tenant}-{ingestion_date}-{uuid4().hex[:8]}"
    state = await _run_graph(
        thread_id,
        initial_state(
            ingestion_date,
            tenant=tenant,
            username=username,
            max_architecture_pending_questions=max_questions_per_stage,
            max_data_contract_pending_questions=max_questions_per_stage,
            # This scopes the run to exactly the files just uploaded. A second,
            # unrelated upload that happens to reuse today's date must never get
            # pooled with this one.
            source_components=result.files_ingested,
            persist=False,
        ),
    )
    return _to_response(thread_id, ingestion_date, result.files_ingested, state)


@router.post("/transcriptions/resume", response_model=TranscriptionResponse)
async def resume_transcription(request: ResumeRequest) -> TranscriptionResponse:
    """Answers a paused graph run's pending questions and continues it. This is the
    same mechanism `ask_human` always uses, whether the pause came from the classify
    stage or a Boss escalation (see `agents/graph.py::ask_human`'s own docstring).

    `expected_tenant` comes from `request.username`, never from `request.thread_id` itself —
    see `_resume_graph`'s own docstring for why a thread_id alone must never be trusted as
    proof of tenant ownership."""
    tenant = _tenant_for_username(request.username)
    state = await _resume_graph(request.thread_id, request.answers, expected_tenant=tenant)
    return _to_response(request.thread_id, state.get("ingestion_date", ""), [], state)


# --- Regenerate with feedback, then finalize (accept) -------------------------------------


class RegenerateRequest(BaseModel):
    source_component: str
    feedback: str
    username: str
    current_document: str


class RegenerateResponse(BaseModel):
    document: str
    score: int
    unresolved_points: list[str]


@router.post("/transcriptions/regenerate", response_model=RegenerateResponse)
async def regenerate_document(
    request: RegenerateRequest, db: AsyncSession = Depends(get_session)
) -> RegenerateResponse:
    """Re-drafts one source's ADR"""
    tenant = _tenant_for_username(request.username)
    bind_tenant(tenant)
    transcript_text = await bronze_content_for_source(db, tenant, request.source_component)
    if not transcript_text:
        raise HTTPException(
            status_code=404, detail=f"No bronze content found for {request.source_component!r}"
        )

    qa_pairs = await qa_pairs_for_source(db, tenant, request.source_component)
    qa_pairs = [*qa_pairs, {"question": "Reviewer feedback on the previous draft", "answer": request.feedback}]

    previous_diagram = own_previous_architecture_diagram(request.current_document)

    messages = build_adr_generation_prompt(transcript_text, qa_pairs, previous_diagram)
    response = await llm_complete(messages)
    document = response.choices[0].message.content

    clarification_items = [
        {"target": request.source_component, "question": qa["question"], "answer": qa["answer"]}
        for qa in qa_pairs
    ]
    critic_messages = build_critic_prompt(document, transcript_text, clarification_items)
    critic_response = await llm_complete(critic_messages, response_format=CritiqueResult)
    critique = CritiqueResult.model_validate(load_json_response(critic_response.choices[0].message.content))

    # This is stamped AFTER the Critic has already reviewed `document`
    document = insert_source_line(insert_authors_line(document, request.username), request.source_component)

    return RegenerateResponse(
        document=document, score=critique.completeness_score, unresolved_points=critique.unresolved_points
    )


class AskMoreRequest(BaseModel):
    username: str
    source_component: str
    max_questions_per_stage: int
    current_document: str
    feedback: str = ""


class AskMoreResponse(BaseModel):
    questions: list[str]


@router.post("/transcriptions/ask-more", response_model=AskMoreResponse)
async def ask_more_questions(request: AskMoreRequest, db: AsyncSession = Depends(get_session)) -> AskMoreResponse:
    """Drafts one more round of clarification questions for an already-drafted ADR by
    re-running the same question-generation + classification standalone (no Boss, no
    persistence), grounded on the reviewer's current draft plus any typed feedback rather
    than a DB re-fetch, and deduped/capped with the same `_top_questions` helper — the
    frontend answers them by folding the Q&A into a feedback string for
    `/transcriptions/regenerate`."""
    tenant = _tenant_for_username(request.username)
    bind_tenant(tenant)
    transcript_text = await bronze_content_for_source(db, tenant, request.source_component)
    if not transcript_text:
        raise HTTPException(
            status_code=404, detail=f"No bronze content found for {request.source_component!r}"
        )
    if request.feedback.strip():
        transcript_text = f"{transcript_text}\n\n[REVIEWER FEEDBACK ON THE CURRENT DRAFT]\n{request.feedback}"
    bronze_ingestion_date = await bronze_ingestion_date_for_source(db, tenant, request.source_component)
    ingestion_date_str = (bronze_ingestion_date or date.today()).strftime("%Y%m%d")
    bronze_rows = [{"source_component": request.source_component, "content": transcript_text}]

    known_architecture = previous_architecture_context(request.current_document)

    architecture_result = await generate_architecture_questions_for_batch(
        ingestion_date_str, bronze_rows, architecture_diagram=known_architecture
    )
    # `generate_data_contract_questions_for_batch` takes `MentionedDataContractItem`,
    # a plain dict/TypedDict. `architecture_result.mentioned_data_contracts` is a list
    # of Pydantic `MentionedDataContract` objects instead. So this must call
    # `.model_dump()` on each one first. This is the same conversion
    # `agents.graph.generate_architecture_questions` already does before writing them
    # into graph state. Skipping it crashes downstream with
    # `'MentionedDataContract' object is not subscriptable`, the moment a real
    # contract was identified.
    contract_result = await generate_data_contract_questions_for_batch(
        ingestion_date_str,
        bronze_rows,
        [c.model_dump() for c in architecture_result.mentioned_data_contracts],
    )
    drafted = list(architecture_result.questions) + list(contract_result.questions)
    if not drafted:
        return AskMoreResponse(questions=[])

    classification_messages = build_classification_prompt([q.model_dump() for q in drafted], transcript_text)
    classification_response = await llm_complete(classification_messages, response_format=ClassificationResult)
    classification = ClassificationResult.model_validate(
        load_json_response(classification_response.choices[0].message.content)
    )

    by_id = {q.id: q for q in drafted}
    needs_clarification: list[str] = []
    scopes: dict[str, str] = {}
    for c in classification.classifications:
        question = by_id.get(c.id)
        if question is None or c.status != "needs_clarification":
            continue
        needs_clarification.append(question.question)
        scopes[question.question] = question.scope

    pending = _top_questions(
        needs_clarification,
        request.max_questions_per_stage,
        request.max_questions_per_stage,
        scopes=scopes,
    )
    return AskMoreResponse(questions=pending)


class FinalizeRequest(BaseModel):
    username: str
    source_component: str
    content: str


class FinalizeResponse(BaseModel):
    version: int


@router.post("/transcriptions/finalize", response_model=FinalizeResponse)
async def finalize_document(request: FinalizeRequest, db: AsyncSession = Depends(get_session)) -> FinalizeResponse:
    """"Publish": persists `content` as this source's next SilverDocument version (a no-op
    if byte-identical to the last one) and immediately runs Gold extraction on it, falling
    back to `BronzeDocument`'s own `ingestion_date` on this source's first-ever publish, or a
    404 if there's no Bronze content either."""
    tenant = _tenant_for_username(request.username)
    bind_tenant(tenant)
    latest_row = (
        await db.execute(
            select(SilverDocument)
            .where(
                SilverDocument.tenant == tenant,
                SilverDocument.source_component == request.source_component,
            )
            .order_by(SilverDocument.version.desc())
            .limit(1)
        )
    ).scalars().first()
    if latest_row is not None:
        ingestion_date = latest_row.ingestion_date
    else:
        ingestion_date = await bronze_ingestion_date_for_source(db, tenant, request.source_component)
        if ingestion_date is None:
            raise HTTPException(
                status_code=404,
                detail=f"No bronze content found for {request.source_component!r} — upload it first",
            )

    version = await _persist_document_version(
        db, ingestion_date, request.source_component, request.content, [], [], tenant
    )
    await db.commit()

    await db.execute(
        delete(SilverChunk).where(
            SilverChunk.tenant == tenant,
            SilverChunk.source_component == request.source_component,
            SilverChunk.version == version,
        )
    )
    [embedding] = await asyncio.to_thread(embed, [request.content])
    db.add(
        SilverChunk(
            tenant=tenant,
            ingestion_date=ingestion_date,
            source_component=request.source_component,
            version=version,
            content=request.content,
            embedding=embedding,
        )
    )
    await db.commit()

    await gold.extract_and_persist_gold_facts(
        db, request.content, request.source_component, version, ingestion_date, tenant=tenant
    )
    await db.commit()

    return FinalizeResponse(version=version)


# --- Architecture history ------------------------------------------------------------------


class ArchitectureHistoryGoldEntity(BaseModel):
    entity_type: str
    canonical_name: str
    operation: str
    version: int


class ArchitectureHistoryAdr(BaseModel):
    source_component: str
    version: int
    ingestion_date: str
    content: str
    authored_by: str
    created_at: str
    content_hash: str
    gold_entities: list[ArchitectureHistoryGoldEntity]


class ArchitectureHistoryResponse(BaseModel):
    diagram: str
    adrs: list[ArchitectureHistoryAdr]


@router.get("/architecture-history", response_model=ArchitectureHistoryResponse)
async def architecture_history(
    username: str, db: AsyncSession = Depends(get_session)
) -> ArchitectureHistoryResponse:
    """Backs the "Architecture history" tab"""
    tenant = _tenant_for_username(username)
    bind_tenant(tenant)
    docs = (
        await db.execute(
            select(SilverDocument).where(SilverDocument.tenant == tenant).order_by(SilverDocument.id.desc())
        )
    ).scalars().all()
    diagram = await gold.current_architecture_diagram(db, tenant=tenant)
    adrs = []
    for doc in docs:
        entities = await gold.gold_entities_for_adr(db, doc.source_component, doc.version, tenant=tenant)
        adrs.append(
            ArchitectureHistoryAdr(
                source_component=doc.source_component,
                version=doc.version,
                ingestion_date=doc.ingestion_date.isoformat(),
                content=doc.content,
                authored_by=doc.authored_by,
                created_at=doc.created_at.isoformat(),
                content_hash=doc.content_hash,
                gold_entities=[
                    ArchitectureHistoryGoldEntity(
                        entity_type=entity.entity_type,
                        canonical_name=entity.canonical_name,
                        operation=entity.operation,
                        version=entity.version,
                    )
                    for entity in entities
                ],
            )
        )
    return ArchitectureHistoryResponse(diagram=diagram, adrs=adrs)


# --- Chat with RAG -------------------------------------------------------------------------


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    username: str
    question: str
    k: int = 8
    history: list[ChatMessage] = []


class ChatResponse(BaseModel):
    answer: str
    retrieved: list[dict]
    citations: list[dict] = []


def _contextualize_question(question: str, history: list[ChatMessage]) -> str:
    """Prefixes `question` with the last `gold.MAX_HISTORY_MESSAGES` turns of `history`, so a
    follow-up that only makes sense in context ("And who approved it?") still retrieves the
    right rows. Used ONLY for retrieval targeting (entity matching, embedding, lexical search)
    — the answer itself is still generated strictly from retrieved Gold facts, via `history`
    passed separately to `gold.answer_question`/`gold.answer_evolution_question`. See
    `.tmp/advanced_techniques.md` §8."""
    if not history:
        return question
    trimmed = history[-gold.MAX_HISTORY_MESSAGES :]
    lines = "\n".join(f"{m.role.capitalize()}: {m.content}" for m in trimmed)
    return f"{lines}\n{question}"


# Matches a reference to one SPECIFIC numbered version ("version 1", "v4", "v.2") — deliberately
# NOT just the bare word "version" (that alone would hijack an ordinary "what version is X on"
# question, which the normal top-k path with its `[latest version]` tag already answers fine).
_SPECIFIC_VERSION_PATTERN = re.compile(r"\bv(?:ersion)?\.?\s*(\d+)\b", re.IGNORECASE)


async def _answer_specific_version_question(
    db: AsyncSession, tenant: str, request: ChatRequest, history: list[tuple[str, str]]
) -> ChatResponse | None:
    """Handles "what were X's facts in version N" — a real, reported gap: `top_k_gold_evolution`
    only ever returns the LATEST version per entity (by design, for "current state" questions),
    so a question pinned to an explicit past version got answered from the wrong version's data
    with no way to reach the right one, even though `entity_history` already had it. This is
    also not what `is_evolution_question`'s full-narrative path is for — that narrates the whole
    timeline; this answers one specific, named snapshot the same way `answer_question` answers
    any other row, `_payload_detail` (input/output contracts, dependencies) included.

    Returns `None` (never a `ChatResponse`) whenever this isn't actually a pinned-version
    question — no version number named, or no known entity named — so `chat` falls through to
    its normal branches exactly as before."""
    version_match = _SPECIFIC_VERSION_PATTERN.search(request.question)
    if version_match is None:
        return None
    match = await gold.find_entity_by_name_in_text(db, request.question, tenant=tenant)
    if match is None:
        return None

    entity_type, entity_id, matched_alias = match
    requested_version = int(version_match.group(1))
    versions = await gold.entity_history(db, entity_type, entity_id, tenant=tenant)
    target_row = next((row for row in versions if row.version == requested_version), None)
    if target_row is None:
        available = ", ".join(str(row.version) for row in versions) or "none"
        return ChatResponse(
            answer=f"{matched_alias} has no version {requested_version}. Versions on record: {available}.",
            retrieved=[],
        )

    latest = await gold.latest_versions(db, [target_row])
    result = await gold.answer_question(db, request.question, [target_row], latest, history=history)
    retrieved = [
        {
            "entity_type": target_row.entity_type,
            "canonical_name": target_row.canonical_name,
            "version": target_row.version,
            "operation": target_row.operation,
        }
    ]
    return ChatResponse(answer=result.answer, retrieved=retrieved, citations=[c.model_dump() for c in result.citations])


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, db: AsyncSession = Depends(get_session)) -> ChatResponse:
    """Answers a chat question from Gold:

    - Anything `Publish` has ever persisted for this tenant is queryable right away — no
      separate approval step.
    - If the question pins one explicit version number ("...in version 1", "v4") to a known
      entity, it answers from exactly that version's own row — never the latest — see
      `_answer_specific_version_question`.
    - If the question names a known component/contract and asks about its history, it
      returns that entity's full version history in chronological order, instead of a
      similarity search.
    - Every other question uses hybrid retrieval: vector similarity combined with a lexical
      search, so an exact name or acronym is never missed.
    - Recent chat history is used only to resolve what a follow-up question refers to — never
      as a source of facts.
    - `tenant` always comes from the logged-in username, never from the request, so one
      tenant can never read another tenant's Gold facts."""
    tenant = _tenant_for_username(request.username)
    bind_tenant(tenant)

    history = [(m.role, m.content) for m in request.history]
    contextualized_question = _contextualize_question(request.question, request.history)

    # Checked first, and only against `request.question` (never `contextualized_question`, same
    # reasoning as the evolution check right below): a question pinned to one explicit version
    # number is unambiguous on its own and must never depend on what an earlier turn said.
    specific_version_response = await _answer_specific_version_question(db, tenant, request, history)
    if specific_version_response is not None:
        return specific_version_response

    # Deliberately `request.question` here, never `contextualized_question`: both checks below
    # must react only to what THIS turn actually asks. `contextualized_question` prefixes prior
    # turns (including the assistant's own past answers) onto the text, so a marker word like
    # "historically" or "timeline" appearing in an EARLIER reply would otherwise flip
    # `is_evolution_question` to True for an unrelated follow-up, and `find_entity_by_name_in_text`
    # (longest-alias-wins) could then match some OTHER entity named in that stale history instead
    # of the one this question actually names — a real, reproduced bug, not a hypothetical one.
    # `contextualized_question` still feeds the embedding/lexical retrieval below, where
    # resolving a pronoun-style follow-up ("and who approved it?") is exactly the point.
    if gold.is_evolution_question(request.question):
        match = await gold.find_entity_by_name_in_text(db, request.question, tenant=tenant)
        if match is not None:
            entity_type, entity_id, matched_alias = match
            rows = await gold.entity_history(db, entity_type, entity_id, tenant=tenant)
            result = await gold.answer_evolution_question(request.question, matched_alias, rows, history=history)
            retrieved = [
                {
                    "entity_type": row.entity_type,
                    "canonical_name": row.canonical_name,
                    "version": row.version,
                    "operation": row.operation,
                    "authored_by": row.authored_by,
                    "ingestion_date": row.ingestion_date.isoformat(),
                }
                for row in rows
            ]
            return ChatResponse(
                answer=result.answer,
                retrieved=retrieved,
                citations=[c.model_dump() for c in result.citations],
            )

    vector = await gold.embed_question(contextualized_question)

    async def _retrieve(max_distance: float | None) -> list[GoldEvolution]:
        # `rerank`/`expand` both cost one extra model call per question, so both stay off here
        # by default, the same way `rerank` already did before `expand` existed — `--rerank`/
        # `--expand` on `scripts/chat_gold.py` are where either gets exercised experimentally.
        return await gold.top_k_gold_evolution(
            db,
            vector,
            k=request.k,
            tenant=tenant,
            mode="hybrid",
            question_text=contextualized_question,
            max_distance=max_distance,
        )

    # Corrective RAG: a strict first pass (`DEFAULT_MAX_DISTANCE`) avoids answering from the
    # least-bad candidate when nothing is actually relevant; if that pass finds nothing, one
    # relaxed retry (no distance filter) catches a real answer that only just missed the
    # strict cutoff, before this gives up honestly. See
    # `agents/stages/gold/retrieval/corrective_rag.py`.
    rows = await gold.retrieve_with_correction(_retrieve, gold.DEFAULT_MAX_DISTANCE)

    if not rows:
        return ChatResponse(answer="No Gold facts were relevant to this question.", retrieved=[])

    latest = await gold.latest_versions(db, rows)
    result = await gold.answer_question(db, request.question, rows, latest, history=history)
    retrieved = [
        {
            "entity_type": row.entity_type,
            "canonical_name": row.canonical_name,
            "version": row.version,
            "operation": row.operation,
        }
        for row in rows
    ]
    return ChatResponse(
        answer=result.answer,
        retrieved=retrieved,
        citations=[c.model_dump() for c in result.citations],
    )


# --- Test monitor --------------------------------------------------------------------------


class LlmCostRow(BaseModel):
    id: int
    method: str
    model: str
    prompt: str
    input_cost: float
    output_cost: float
    created_at: str


class TestMonitorResponse(BaseModel):
    suites: list[dict]
    llm_costs: list[LlmCostRow]


# `test_monitor` below only ever returns this many of the caller's tenant's most recent
# `llm_costs` rows. Every real LLM call writes one row (`llm.router._log_usage`), so an
# unbounded query here would grow without limit as the app keeps running — this caps what one
# Monitor-tab load actually pulls and renders, newest calls first.
_LLM_COSTS_DISPLAY_LIMIT = 200


@router.get("/test-monitor", response_model=TestMonitorResponse)
async def test_monitor(username: str, db: AsyncSession = Depends(get_session)) -> TestMonitorResponse:
    """Backs the "Monitor" tab. Test suite results (`testing_*/output/result.json`) stay a
    global, app-wide view — the ACB golden sets are not tenant data. `llm_costs`, below, is the
    opposite: every real LLM call is tagged with the tenant `bind_tenant` set at that call's
    own request boundary (see `llm.router.complete`'s own docstring), so this endpoint filters
    to the CALLER's own tenant only, the same isolation `architecture_history`/`chat` already
    enforce — one tenant must never see how much another tenant's usage cost. Gated only by
    `start_test_mode`; the endpoint itself stays reachable, `start_test_mode` only hides the
    tab that calls it."""
    if not settings.start_test_mode:
        raise HTTPException(status_code=404, detail="Test monitor is disabled (start_test_mode=False)")
    tenant = _tenant_for_username(username)

    suites = []
    for result_path in sorted(Path(".").glob("testing_*/output/result.json")):
        suites.append({"suite": result_path.parent.parent.name, "results": json.loads(result_path.read_text())})

    rows = (
        await db.execute(
            select(LlmCost)
            .where(LlmCost.tenant == tenant)
            .order_by(LlmCost.created_at.desc())
            .limit(_LLM_COSTS_DISPLAY_LIMIT)
        )
    ).scalars().all()
    llm_costs = [
        LlmCostRow(
            id=row.id,
            method=row.method,
            model=row.model,
            prompt=row.prompt,
            input_cost=row.input_cost,
            output_cost=row.output_cost,
            created_at=row.created_at.isoformat(),
        )
        for row in rows
    ]

    return TestMonitorResponse(suites=suites, llm_costs=llm_costs)
