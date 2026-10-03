"""Every endpoint the Streamlit app calls. One router for all four tabs — see the
README's "User interface" section.

Auth (`/login`) checks `settings.frontend_users`, a fixed list with no user table (see
`FrontendUser` in `app/config.py`). `tenant` keeps data separate between logins. There
is no session token: the frontend holds `{username, tenant}` in `st.session_state` and
sends `tenant` back on every call.
"""

import asyncio
import json
import re
import time
from collections import defaultdict, deque
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

from agents.graph import (
    _persist_document_version,
    _top_questions,
    build_graph,
    checkpointer_dsn,
    strip_downgrade_markers,
)
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
from agents.stages.adr_generation.service import fix_diagram_class_references, own_previous_architecture_diagram
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


# Per-tenant sliding window of call timestamps, for `_enforce_llm_rate_limit`. In-process
# only: it resets on restart and does not share across workers. A multi-worker deployment
# needs this moved to Postgres or Redis.
_llm_call_log: dict[str, deque[float]] = defaultdict(deque)


def _enforce_llm_rate_limit(tenant: str) -> None:
    """Cost guardrail for the three endpoints with a real paid LLM call per request
    (`regenerate_document`, `ask_more_questions`, `finalize_document`). Rejects a request once
    `tenant` exceeds `settings.max_llm_calls_per_window` calls within
    `settings.llm_rate_limit_window_seconds`. Call this right after `_tenant_for_username`."""
    now = time.monotonic()
    window = _llm_call_log[tenant]
    cutoff = now - settings.llm_rate_limit_window_seconds
    while window and window[0] < cutoff:
        window.popleft()
    if len(window) >= settings.max_llm_calls_per_window:
        raise HTTPException(
            status_code=429,
            detail=(
                f"Too many AI requests for this tenant — max {settings.max_llm_calls_per_window} "
                f"per {settings.llm_rate_limit_window_seconds}s. Wait and try again."
            ),
        )
    window.append(now)


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
    # Maps source_component to the Critic's completeness_score (0-100).
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
    """Checks that this `thread_id`'s saved tenant matches `expected_tenant` before resuming
    it. `thread_id` is a guessable string, not a secret. This check stops a request from
    reading another tenant's paused draft."""
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
    """Ingests every uploaded file into Bronze, capped at `settings.max_upload_file_bytes`.
    Runs the Silver clarification loop with `persist=False`, so the result stays a draft
    until "Publish". Pauses with `status: "pending_review"` (resume via `/resume`) if a human
    must answer up to `max_questions_per_stage` questions per stage."""
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
            # Scopes the run to only the files just uploaded. A second, unrelated
            # upload on the same date must not pool with this one.
            source_components=result.files_ingested,
            persist=False,
        ),
    )
    return _to_response(thread_id, ingestion_date, result.files_ingested, state)


@router.post("/transcriptions/resume", response_model=TranscriptionResponse)
async def resume_transcription(request: ResumeRequest) -> TranscriptionResponse:
    """Answers a paused graph run's pending questions and continues it. Same mechanism as
    `ask_human` (see `agents/graph.py::ask_human`), for a pause from the classify stage or a
    Boss escalation.

    `expected_tenant` comes from `request.username`, never from `request.thread_id` — see
    `_resume_graph` for why a thread_id alone is not proof of tenant ownership."""
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
    _enforce_llm_rate_limit(tenant)
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
    document = fix_diagram_class_references(response.choices[0].message.content)

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
    """Drafts one more round of clarification questions for an already-drafted ADR. Re-runs
    question-generation and classification standalone, with no Boss step and no persistence.
    Grounds the questions on the reviewer's current draft and typed feedback, not a DB
    re-fetch. Dedupes and caps the result with `_top_questions`. The frontend folds the
    answers into a feedback string for `/transcriptions/regenerate`."""
    tenant = _tenant_for_username(request.username)
    bind_tenant(tenant)
    _enforce_llm_rate_limit(tenant)
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
    # `generate_data_contract_questions_for_batch` takes plain dicts
    # (`MentionedDataContractItem`), but `mentioned_data_contracts` holds Pydantic
    # `MentionedDataContract` objects. Call `.model_dump()` on each first — the same
    # conversion `agents.graph.generate_architecture_questions` does. Skipping this
    # crashes with `'MentionedDataContract' object is not subscriptable`.
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
    """"Publish": persists `content` as this source's next SilverDocument version (a no-op if
    byte-identical to the last one), then runs Gold extraction on it. Falls back to
    `BronzeDocument`'s own `ingestion_date` on this source's first publish, or 404 if there is
    no Bronze content either.

    `strip_downgrade_markers` runs on `request.content` first. The reviewed draft still
    shows `_DOWNGRADE_MARKER` annotations (`agents.graph.boss_decide`'s "unverified claim"
    flag). Once published, that annotation has done its job and must not stay in the ADR's
    record. Stripping it here means every later reader sees the same clean text the human
    accepted."""
    tenant = _tenant_for_username(request.username)
    bind_tenant(tenant)
    _enforce_llm_rate_limit(tenant)
    content = strip_downgrade_markers(request.content)
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
        db, ingestion_date, request.source_component, content, [], [], tenant
    )
    await db.commit()

    await db.execute(
        delete(SilverChunk).where(
            SilverChunk.tenant == tenant,
            SilverChunk.source_component == request.source_component,
            SilverChunk.version == version,
        )
    )
    [embedding] = await asyncio.to_thread(embed, [content])
    db.add(
        SilverChunk(
            tenant=tenant,
            ingestion_date=ingestion_date,
            source_component=request.source_component,
            version=version,
            content=content,
            embedding=embedding,
        )
    )
    await db.commit()

    await gold.extract_and_persist_gold_facts(
        db, content, request.source_component, version, ingestion_date, tenant=tenant
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
    # Mermaid code from `gold.current_architecture_diagram_interactive`. Rendered by
    # `streamlit_mermaid_interactive`, not `st.mermaid_chart` — see that function's docstring.
    diagram: str
    # Maps a node's visible label to a "{source_component}::{source_adr_version}" string. The
    # frontend component IDs a clicked node by its rendered text, not a Mermaid node id.
    diagram_entity_mapping: dict[str, str]
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
    diagram, diagram_entity_mapping = await gold.current_architecture_diagram_interactive(db, tenant=tenant)
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
    return ArchitectureHistoryResponse(
        diagram=diagram, diagram_entity_mapping=diagram_entity_mapping, adrs=adrs
    )


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
    diagram: str | None = None
    diagram_sources: list[dict] = []


def _contextualize_question(question: str, history: list[ChatMessage]) -> str:
    """Prefixes `question` with the last `gold.MAX_HISTORY_MESSAGES` turns of `history`. This
    lets a context-only follow-up ("And who approved it?") still retrieve the right rows.
    Used only for retrieval: entity matching, embedding, lexical search. The answer itself
    still comes only from retrieved Gold facts; `history` reaches
    `gold.answer_question`/`gold.answer_evolution_question` separately."""
    if not history:
        return question
    trimmed = history[-gold.MAX_HISTORY_MESSAGES :]
    lines = "\n".join(f"{m.role.capitalize()}: {m.content}" for m in trimmed)
    return f"{lines}\n{question}"


async def _diagram_response_fields(db: AsyncSession, tenant: str, rows: list, citations: list) -> dict:
    """Scopes `build_relationship_diagram` to the component(s) an answer actually CITED, never
    a row that was only retrieved. Each of `chat`'s three response branches calls this once,
    right before building its `ChatResponse`. Returns `{}` when there is nothing to show, so a
    call site can write `ChatResponse(..., **await _diagram_response_fields(...))` with no
    None-check."""
    cited_keys = {(c.entity_type, c.entity_id, c.version) for c in citations}
    cited_rows = [row for row in rows if (row.entity_type, row.entity_id, row.version) in cited_keys]
    built = await gold.build_relationship_diagram(db, cited_rows, tenant=tenant)
    if built is None:
        return {}
    diagram, focal_rows = built
    return {
        "diagram": diagram,
        "diagram_sources": [
            {
                "canonical_name": row.canonical_name,
                "source_component": row.source_component,
                "source_adr_version": row.source_adr_version,
            }
            for row in focal_rows
        ],
    }


# Matches one specific numbered version ("version 1", "v4", "v.2"). Not the bare word
# "version": that would hijack an ordinary "what version is X on" question, which the
# normal top-k path already answers.
_SPECIFIC_VERSION_PATTERN = re.compile(r"\bv(?:ersion)?\.?\s*(\d+)\b", re.IGNORECASE)


async def _answer_specific_version_question(
    db: AsyncSession, tenant: str, request: ChatRequest, history: list[tuple[str, str]]
) -> ChatResponse | None:
    """Handles "what were X's facts in version N". `top_k_gold_evolution` only returns the
    latest version per entity, by design, so a question pinned to a past version needs this
    path instead. Not the same as `is_evolution_question`'s full-timeline path: this answers
    one named snapshot, the same way `answer_question` answers any other row.

    Returns `None` when the question names no version number or no known entity, so `chat`
    falls through to its normal branches."""
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
    return ChatResponse(
        answer=result.answer,
        retrieved=retrieved,
        citations=[c.model_dump() for c in result.citations],
        **await _diagram_response_fields(db, tenant, [target_row], result.citations),
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, db: AsyncSession = Depends(get_session)) -> ChatResponse:
    """Answers a chat question from Gold:

    - Anything `Publish` has persisted for this tenant is queryable right away. No separate
      approval step.
    - A question pinned to one version number ("in version 1", "v4") answers from that row,
      never the latest — see `_answer_specific_version_question`.
    - A question about a known entity's history returns its full version history in order,
      not a similarity search.
    - Every other question uses hybrid retrieval: vector similarity plus lexical search, so
      an exact name or acronym is never missed.
    - A cited component adds a small Mermaid `diagram` of it and its direct neighbors
      (`diagram_sources` names the ADR(s) to link to) — see `_diagram_response_fields`.
      `None` when nothing was cited, or a cited component has no relationships.
    - Chat history only resolves what a follow-up refers to. It is never a source of facts.
    - `tenant` comes from the logged-in username, never the request, so a tenant can never
      read another tenant's Gold facts."""
    tenant = _tenant_for_username(request.username)
    bind_tenant(tenant)

    history = [(m.role, m.content) for m in request.history]
    contextualized_question = _contextualize_question(request.question, request.history)

    # Checks `request.question`, never `contextualized_question`: a pinned version number is
    # unambiguous on its own and must not depend on an earlier turn.
    specific_version_response = await _answer_specific_version_question(db, tenant, request, history)
    if specific_version_response is not None:
        return specific_version_response

    # Checks `request.question`, never `contextualized_question`, for the same reason: a
    # marker word in an EARLIER reply (e.g. "historically") could otherwise flip
    # `is_evolution_question` for an unrelated follow-up, and match the wrong entity — a real,
    # reproduced bug. `contextualized_question` still feeds retrieval below, where resolving a
    # follow-up ("and who approved it?") is the point.
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
                **await _diagram_response_fields(db, tenant, rows, result.citations),
            )

    vector = await gold.embed_question(contextualized_question)

    async def _retrieve(max_distance: float | None) -> list[GoldEvolution]:
        # `rerank`/`expand` each cost one extra model call per question, so both stay off
        # here by default. `scripts/chat_gold.py --rerank`/`--expand` exercise them instead.
        return await gold.top_k_gold_evolution(
            db,
            vector,
            k=request.k,
            tenant=tenant,
            mode="hybrid",
            question_text=contextualized_question,
            max_distance=max_distance,
        )

    # Corrective RAG: a strict first pass avoids answering from a weak candidate. One
    # relaxed retry, with no distance filter, runs only if that pass finds nothing. See
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
        **await _diagram_response_fields(db, tenant, rows, result.citations),
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


# Caps how many of the caller's `llm_costs` rows `test_monitor` returns. Every LLM call
# writes one row, so an unbounded query would grow without limit. Newest calls first.
_LLM_COSTS_DISPLAY_LIMIT = 200


@router.get("/test-monitor", response_model=TestMonitorResponse)
async def test_monitor(username: str, db: AsyncSession = Depends(get_session)) -> TestMonitorResponse:
    """Backs the "Monitor" tab. Test suite results are a global, app-wide view: the golden
    sets are not tenant data. `llm_costs` is the opposite — each row is tagged with the
    tenant `bind_tenant` set at that call (see `llm.router.complete`). This endpoint filters
    to the caller's own tenant, like `architecture_history`/`chat` already do. The route
    stays registered either way; `start_test_mode=False` makes it 404 on every call."""
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
