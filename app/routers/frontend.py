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
from db.models import LlmCost, SilverChunk, SilverDocument
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
    """The one place a `username` turns into the tenant a request is allowed to touch. Every
    endpoint below that used to take a client-supplied `tenant` field alongside `username` now
    calls this instead, and no longer accepts `tenant` as its own independent input at all.

    Before this, a request's `tenant` and `username` were two unrelated fields — nothing
    stopped a caller from sending a real `username` next to ANY `tenant` string, including one
    that belongs to a different login entirely. That defeated the isolation
    `FrontendUser.tenant`'s own docstring already claims: "every query the frontend makes also
    filters by tenant" is only true if the tenant used really is that user's own. This makes it
    true: the tenant a request acts on is always looked up from `settings.frontend_users`,
    never taken as a bare string the caller asserts.

    This does not add real per-request authentication — this module's own docstring already
    documents that there is no session token, and that is unchanged here. A caller still only
    needs to know a valid username to act as it, the same trust level `uploaded_by`/
    `authored_by` already relied on. What changes is narrower and specific: which TENANT'S data
    a request can reach is no longer separately choosable by the caller — it is exactly the one
    tenant that username is actually registered under."""
    for user in settings.frontend_users:
        if user.username == username:
            return user.tenant
    raise HTTPException(status_code=401, detail="Unknown user")


class ConfigResponse(BaseModel):
    start_test_mode: bool


@router.get("/config", response_model=ConfigResponse)
async def config() -> ConfigResponse:
    """Lets the frontend decide whether to show the Test monitor tab at all. This way
    the frontend does not have to pay `/test-monitor`'s own file-reading cost just to
    find out."""
    return ConfigResponse(start_test_mode=settings.start_test_mode)


# --- Input transcription -------------------------------------------------------------------


class TranscriptionResponse(BaseModel):
    ingestion_date: str
    files_ingested: list[str] = []
    status: Literal["completed", "pending_review"]
    thread_id: str
    pending_questions: list[str] = []
    documents: dict[str, str] = {}
    # This maps source_component to the Critic's completeness_score (0-100) and
    # unresolved_points. See the COMPLETENESS SCORING section in
    # prompts/adr_critic/critic.jinja. It stays empty until `status` is "completed". A
    # paused run has no finished document yet to score.
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
    """`thread_id` is a plain string a client can type, log, or guess
    (`f"frontend-{tenant}-{ingestion_date}-{uuid}"`, see `upload_transcription`) — nothing about
    it is a secret or a credential. Without a check here, a request that happened to know or
    guess another tenant's `thread_id` could resume THAT tenant's paused clarification session,
    reading back its draft ADR content in the response. This is exactly the "only same-tenant
    users can see their ADRs" guardrail, applied to a draft still mid-clarification, not just to
    a published one.

    `graph.aget_state` reads the checkpoint's own last-saved `SilverState` without resuming
    anything — a nonexistent `thread_id` comes back with `values == {}`, which never equals a
    real tenant string, so it fails this same check rather than needing special-casing."""
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
    """Ingests every uploaded file into Bronze, then runs the Silver clarification loop
    on it. This does "to be processed and clarified" in one step, matching the tab's
    own description. A transcript that needs human input pauses here
    (`status: "pending_review"`). Answer it through `/resume`.

    This runs with `persist=False`. So the ADR this produces, and any later
    "Regenerate ADR" or "Ask me more" refinement of it, is a draft only. Nothing lands
    in `silver_documents` or Gold until the reviewer explicitly clicks "Publish"
    (`finalize_document`). See `SilverState.persist`'s own docstring for why this run
    must never write those on its own. It only writes its clarification audit trail
    (`write_document` always writes `silver_clarifications` regardless).

    `max_questions_per_stage` caps both the architecture and the data-contract
    pending-question lists at the SAME number. See `agents.state.initial_state` and
    `agents.graph._top_questions`. A value of 4 means at most 4 + 4 = 8 questions in
    total.

    `tenant` is never taken from the request. It is `_tenant_for_username(username)` — the
    logged-in user IS the tenant this upload belongs to, not a separately chosen value. The
    same `username` also becomes `bronze_documents.uploaded_by` and, later,
    `silver_documents.authored_by` (via `insert_authors_line`): the person running this upload
    is the author of the input and, once published, of the ADR.

    Each file is capped at `settings.max_upload_file_bytes` (see that setting's own comment).
    `f.read(limit + 1)` reads at most one byte past the limit — enough to tell "too big" apart
    from "exactly at the limit" — without ever pulling an arbitrarily large file fully into
    memory just to measure it. This FastAPI/Starlette version's `UploadFile` has no `.size`
    attribute to check up front instead."""
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
    # This is the draft exactly as the reviewer is looking at it right now. See
    # `AskMoreRequest.current_document`'s own docstring for why this must be the
    # client's copy, not a DB re-fetch. The initial upload now runs with
    # `persist=False` (`SilverState.persist`). So there is often no `SilverDocument`
    # row at all yet to read a "previous architecture" from. And when one DOES exist,
    # it may belong to an earlier, unrelated PUBLISHED change for this same
    # source_component, never to this session's own current draft.
    current_document: str


class RegenerateResponse(BaseModel):
    document: str
    score: int
    unresolved_points: list[str]


@router.post("/transcriptions/regenerate", response_model=RegenerateResponse)
async def regenerate_document(
    request: RegenerateRequest, db: AsyncSession = Depends(get_session)
) -> RegenerateResponse:
    """Re-drafts one source's ADR, folding in the reviewer's free-text feedback. This is
    the same Actor call `synthesize_document` makes inside the graph, plus the same
    Critic call for an updated completeness score, run standalone here: no Boss, no
    persistence. The reviewer is already the human steering this draft directly, by
    giving feedback. Nothing is written to Silver or Gold until they call
    `/transcriptions/finalize` on whichever draft they accept.

    `tenant` comes from `_tenant_for_username(request.username)`, not a client-supplied field —
    see that function's own docstring. This is also what stops a request from reading another
    tenant's `bronze_documents`/clarifications by source name alone."""
    tenant = _tenant_for_username(request.username)
    bind_tenant(tenant)
    transcript_text = await bronze_content_for_source(db, tenant, request.source_component)
    if not transcript_text:
        raise HTTPException(
            status_code=404, detail=f"No bronze content found for {request.source_component!r}"
        )

    qa_pairs = await qa_pairs_for_source(db, tenant, request.source_component)
    qa_pairs = [*qa_pairs, {"question": "Reviewer feedback on the previous draft", "answer": request.feedback}]

    # This uses §2, not §3, of the reviewer's own current draft. The change itself has
    # not finalized yet. So the previous architecture must stay whatever it already
    # was. It must not shift to this same draft's own just-generated target. See
    # `own_previous_architecture_diagram`.
    previous_diagram = own_previous_architecture_diagram(request.current_document)

    messages = build_adr_generation_prompt(transcript_text, qa_pairs, previous_diagram)
    response = await llm_complete(messages)
    document = response.choices[0].message.content

    # build_critic_prompt expects the richer ClarificationItem shape, keyed by
    # `target`. SilverClarification's own audit trail only ever stored a question and
    # an answer. So `target` is filled with the source name here, just to satisfy that
    # shape. The Critic only uses it for display, never to resolve anything.
    clarification_items = [
        {"target": request.source_component, "question": qa["question"], "answer": qa["answer"]}
        for qa in qa_pairs
    ]
    critic_messages = build_critic_prompt(document, transcript_text, clarification_items)
    critic_response = await llm_complete(critic_messages, response_format=CritiqueResult)
    critique = CritiqueResult.model_validate(load_json_response(critic_response.choices[0].message.content))

    # This is stamped AFTER the Critic has already reviewed `document`. See
    # `insert_authors_line`'s own docstring for why doing this any earlier would get
    # the line flagged as an unsupported claim — `insert_source_line` carries the same
    # constraint, for the same reason.
    document = insert_source_line(insert_authors_line(document, request.username), request.source_component)

    return RegenerateResponse(
        document=document, score=critique.completeness_score, unresolved_points=critique.unresolved_points
    )


class AskMoreRequest(BaseModel):
    username: str
    source_component: str
    max_questions_per_stage: int
    # This is the draft exactly as the reviewer is looking at it right now: the
    # original synthesis, or whatever a prior "Regenerate ADR" or "Ask me more" round
    # produced. Nothing about a redraft is ever persisted until "Publish" (see
    # `finalize_document`). So the backend has no other way to know what this ADR
    # currently says. Reading `SilverDocument` here would silently fall back to the
    # FIRST draft, and re-ask questions the reviewer already resolved.
    current_document: str
    # These are free-text notes currently sitting in the feedback box, whether
    # submitted or not. The reviewer may have typed something ("also handle X", "Y is
    # wrong") without clicking "Regenerate ADR" yet. That is still live information
    # this round of questions should account for.
    feedback: str = ""


class AskMoreResponse(BaseModel):
    questions: list[str]


@router.post("/transcriptions/ask-more", response_model=AskMoreResponse)
async def ask_more_questions(request: AskMoreRequest, db: AsyncSession = Depends(get_session)) -> AskMoreResponse:
    """Drafts one fresh round of clarification questions for a draft that already went
    through its first pass. This is the same two-stage generation
    (`generate_architecture_questions` and `generate_data_contract_questions`) that
    runs inside the graph, plus the same classification (`classify_questions`) run,
    called standalone here: no Boss, no persistence, the same reasoning as
    `regenerate_document`. It lets the reviewer pull for more detail with targeted
    questions, instead of writing free-text feedback by hand. The frontend answers
    them and calls `/transcriptions/regenerate` with the Q&A folded into one feedback
    string. This reuses that endpoint rather than duplicating its Actor/Critic call.

    This grounds the new questions on `request.current_document`, the reviewer's own
    current draft, NOT a DB re-fetch (see `AskMoreRequest.current_document`'s own
    docstring for why). It does this through `previous_architecture_context`, using
    its §3/§4, not its §2. See `own_previous_architecture_diagram`'s docstring for why
    that distinction matters elsewhere. It also appends `request.feedback`, when
    given, to the transcript text itself before drafting. A live complaint or extra
    detail the reviewer typed is exactly the kind of thing
    `architecture_questions.jinja` should ground new questions on, the same way it
    would if it had been said in the meeting itself.

    `_top_questions` is `agents.graph`'s own private dedupe-and-cap helper. It is
    reused here, not reimplemented, for the same reason `_persist_document_version` is
    reused by `finalize_document` below: one algorithm, not a second copy that could
    drift from it.

    `tenant` comes from `_tenant_for_username(request.username)`, never a client-supplied
    field — see that function's own docstring."""
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
    """"Publish": persists `content`, whichever draft the reviewer is looking at,
    original or regenerated, as this source's next SilverDocument version. This is a
    no-op version bump if `content` is byte-identical to what is already there. It
    also runs Gold extraction on it. That is the whole action: publishing an ADR means
    its facts are in Gold and can be retrieved from Chat immediately, with no separate
    approval step. This reuses `_persist_document_version` (the same
    hash-compare-then-bump `write_document` uses inside the graph) and
    `gold.extract_and_persist_gold_facts` (the same shared body
    `scripts/backfill_gold.py` uses). This is the second and third real caller of
    each, not a third or fourth copy of either.

    This is very often the FIRST time this source gets a `SilverDocument` row at all.
    The initial upload runs with `persist=False` (see `SilverState.persist`), so there
    is usually no existing row yet to read `ingestion_date` from. This falls back to
    `BronzeDocument`'s own `ingestion_date` for exactly that case. Only a source with
    no bronze content either is a real 404 ("nothing was ever uploaded for this
    name").

    `tenant` comes from `_tenant_for_username(request.username)`, never a client-supplied
    field — see that function's own docstring. This is what stops a request from publishing
    into, or reading the version history of, another tenant's `source_component`."""
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
    """Backs the "Architecture history" tab: a read-only view of every ADR this tenant
    has on record, newest first, plus a live Mermaid diagram of Gold's current
    component state (`gold.current_architecture_diagram`, not any single ADR's own
    LLM-drawn diagram).

    This is the main guardrail this whole module exists to enforce: only a user of the SAME
    tenant can see its ADRs. `tenant` comes from `_tenant_for_username(username)`, never a
    client-supplied query parameter — a caller can no longer read another tenant's ADR history
    by simply passing a different `tenant` string, since there is no such parameter to pass
    anymore."""
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


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, db: AsyncSession = Depends(get_session)) -> ChatResponse:
    """Answers from Gold. Every fact `Publish` has ever persisted for this tenant can
    be retrieved immediately, with no separate approval gate.

    This picks between two retrieval strategies, the same choice `scripts/chat_gold.py`'s own
    REPL makes — see that script's module docstring for the full reasoning. In short: a
    question that both names a known component/data contract and reads as asking for its
    history ("how has X evolved over time?" / "qué evolución ha tenido X") gets that one
    entity's COMPLETE version history (`gold.entity_history`), narrated chronologically
    (`gold.answer_evolution_question`) — never embedding similarity, since the entity is
    already known and top-k could otherwise drop an early version. Every other question uses
    hybrid top-k retrieval (`mode="hybrid"`): vector similarity fused with a lexical `ts_rank`
    search via Reciprocal Rank Fusion, so an exact component name/acronym/ODCS field name the
    embedding alone might blur still surfaces — see `top_k_gold_evolution`'s own docstring and
    `.tmp/advanced_techniques.md` §1.

    `request.history` (recent `(role, content)` turns from the client's own chat transcript)
    is used two ways: `_contextualize_question` folds it into the text handed to entity
    matching/embedding/lexical search, so a context-dependent follow-up still resolves to the
    right entity or rows; the raw history is also passed straight through to
    `gold.answer_question`/`gold.answer_evolution_question`, which use it ONLY to resolve what
    the question refers to, never as a source of facts — see `.tmp/advanced_techniques.md` §8.

    `tenant` comes from `_tenant_for_username(request.username)`, never a client-supplied
    field — this is what stops a chat question from ever retrieving another tenant's Gold
    facts, the same guardrail `architecture_history` applies to the ADR history view."""
    tenant = _tenant_for_username(request.username)
    bind_tenant(tenant)

    history = [(m.role, m.content) for m in request.history]
    contextualized_question = _contextualize_question(request.question, request.history)

    if gold.is_evolution_question(contextualized_question):
        match = await gold.find_entity_by_name_in_text(db, contextualized_question, tenant=tenant)
        if match is not None:
            entity_type, entity_id, matched_alias = match
            rows = await gold.entity_history(db, entity_type, entity_id, tenant=tenant)
            answer = await gold.answer_evolution_question(request.question, matched_alias, rows, history=history)
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
            return ChatResponse(answer=answer, retrieved=retrieved)

    vector = await gold.embed_question(contextualized_question)
    rows = await gold.top_k_gold_evolution(
        db, vector, k=request.k, tenant=tenant, mode="hybrid", question_text=contextualized_question
    )

    if not rows:
        return ChatResponse(answer="No Gold facts were relevant to this question.", retrieved=[])

    latest = await gold.latest_versions(db, rows)
    answer = await gold.answer_question(db, request.question, rows, latest, history=history)
    retrieved = [
        {
            "entity_type": row.entity_type,
            "canonical_name": row.canonical_name,
            "version": row.version,
            "operation": row.operation,
        }
        for row in rows
    ]
    return ChatResponse(answer=answer, retrieved=retrieved)


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
