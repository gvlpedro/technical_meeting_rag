"""Every endpoint the Streamlit app in `frontend/` calls — one router, not five, since all of
it exists to serve that one app's four tabs (see README's "User interface" section) and
splitting it further would only scatter one cohesive story across more files.

Auth (`/login`) checks against `settings.frontend_users` — a fixed, hardcoded roster, no user
table (see `FrontendUser`'s own docstring in `app/config.py`). `tenant` is what actually
isolates data between logins; nothing here issues a session token, so the frontend just holds
`{username, tenant}` in its own `st.session_state` after a successful login and sends `tenant`
back on every later call.
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

from agents import gold_service
from agents.graph import _persist_document_version, _top_questions, build_graph, checkpointer_dsn
from agents.prompts import build_adr_generation_prompt, build_classification_prompt, build_critic_prompt
from agents.schemas import ClassificationResult, CritiqueResult
from agents.service import (
    bronze_content_for_source,
    bronze_ingestion_date_for_source,
    generate_architecture_questions_for_batch,
    generate_data_contract_questions_for_batch,
    latest_document_content,
    own_previous_architecture_diagram,
    previous_architecture_context,
    qa_pairs_for_source,
)
from agents.state import initial_state
from agents.template import load_json_response
from app.config import settings
from db.models import SilverChunk, SilverDocument
from db.session import get_session
from ingestion.embedder import embed
from ingestion.service import NoTranscriptsFoundError, ingest_uploaded_files
from llm.router import LLM_USAGE_LOG, bind_tenant
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


class ConfigResponse(BaseModel):
    start_test_mode: bool


@router.get("/config", response_model=ConfigResponse)
async def config() -> ConfigResponse:
    """Lets the frontend decide whether to show the Test monitor tab at all, without paying
    for `/test-monitor`'s own file-reading cost just to find out."""
    return ConfigResponse(start_test_mode=settings.start_test_mode)


# --- Input transcription -------------------------------------------------------------------


class TranscriptionResponse(BaseModel):
    ingestion_date: str
    files_ingested: list[str] = []
    status: Literal["completed", "pending_review"]
    thread_id: str
    pending_questions: list[str] = []
    documents: dict[str, str] = {}
    # source_component -> Critic's completeness_score (0-100) / unresolved_points — see
    # prompts/adr_critic.jinja's COMPLETENESS SCORING section. Empty until `status` is
    # "completed"; a paused run has no finished document yet to score.
    scores: dict[str, int] = {}
    unresolved_points: dict[str, list[str]] = {}


class ResumeRequest(BaseModel):
    thread_id: str
    answers: dict[str, str]


async def _run_graph(thread_id: str, payload: dict) -> dict:
    async with AsyncPostgresSaver.from_conn_string(checkpointer_dsn()) as saver:
        await saver.setup()
        graph = build_graph(saver)
        return await graph.ainvoke(payload, config={"configurable": {"thread_id": thread_id}})


async def _resume_graph(thread_id: str, answers: dict[str, str]) -> dict:
    async with AsyncPostgresSaver.from_conn_string(checkpointer_dsn()) as saver:
        await saver.setup()
        graph = build_graph(saver)
        return await graph.ainvoke(
            Command(resume=answers), config={"configurable": {"thread_id": thread_id}}
        )


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
    tenant: str = Form(...),
    ingestion_date: str = Form(...),
    max_questions_per_stage: int = Form(...),
    files: list[UploadFile] = File(...),
    db: AsyncSession = Depends(get_session),
) -> TranscriptionResponse:
    """Ingests every uploaded file into Bronze, then runs the full Silver+Gold graph on it —
    "to be processed and clarified" in one step, per the tab's own description. A transcript
    that needs human input pauses here (`status: "pending_review"`); answer it via `/resume`.

    `max_questions_per_stage` caps both the architecture and the data-contract pending-
    question lists at the SAME number — see `agents.state.initial_state` and
    `agents.graph._top_questions`. A value of 4 means at most 4 + 4 = 8 questions total.
    """
    payloads = [(f.filename or "upload", await f.read()) for f in files]
    try:
        result = await ingest_uploaded_files(payloads, ingestion_date, tenant, db)
    except (ValueError, NoTranscriptsFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    bind_tenant(tenant)
    thread_id = f"frontend-{tenant}-{ingestion_date}-{uuid4().hex[:8]}"
    state = await _run_graph(
        thread_id,
        initial_state(
            ingestion_date,
            tenant=tenant,
            max_architecture_pending_questions=max_questions_per_stage,
            max_data_contract_pending_questions=max_questions_per_stage,
            # Scope this run to exactly the files just uploaded — a second, unrelated upload
            # that happens to reuse today's date must never get pooled with this one.
            source_components=result.files_ingested,
        ),
    )
    return _to_response(thread_id, ingestion_date, result.files_ingested, state)


@router.post("/transcriptions/resume", response_model=TranscriptionResponse)
async def resume_transcription(request: ResumeRequest) -> TranscriptionResponse:
    """Answers a paused graph run's pending questions and continues it — same mechanism
    `ask_human` always uses, whether the pause came from the classify stage or a Boss
    escalation (see `agents/graph.py::ask_human`'s own docstring)."""
    state = await _resume_graph(request.thread_id, request.answers)
    return _to_response(request.thread_id, state.get("ingestion_date", ""), [], state)


# --- Regenerate with feedback, then finalize (accept) -------------------------------------


class RegenerateRequest(BaseModel):
    tenant: str
    source_component: str
    feedback: str


class RegenerateResponse(BaseModel):
    document: str
    score: int
    unresolved_points: list[str]


@router.post("/transcriptions/regenerate", response_model=RegenerateResponse)
async def regenerate_document(
    request: RegenerateRequest, db: AsyncSession = Depends(get_session)
) -> RegenerateResponse:
    """Re-drafts one source's ADR incorporating the reviewer's free-text feedback — the same
    Actor call `synthesize_document` makes inside the graph, plus the same Critic call for an
    updated completeness score, run standalone here: no Boss, no persistence. The reviewer is
    already the human steering this draft directly by giving feedback; nothing is written to
    Silver/Gold until they call `/transcriptions/finalize` on whichever draft they accept."""
    bind_tenant(request.tenant)
    transcript_text = await bronze_content_for_source(db, request.tenant, request.source_component)
    if not transcript_text:
        raise HTTPException(
            status_code=404, detail=f"No bronze content found for {request.source_component!r}"
        )

    qa_pairs = await qa_pairs_for_source(db, request.tenant, request.source_component)
    qa_pairs = [*qa_pairs, {"question": "Reviewer feedback on the previous draft", "answer": request.feedback}]

    # This draft's own latest SilverDocument row — already persisted by `write_document`
    # before the reviewer ever saw it, since nothing here is approved yet. Its §2 (not §3) is
    # this regeneration's "previous architecture": the change itself hasn't finalized, so the
    # previous architecture must stay whatever it already was, not shift to this same draft's
    # own just-generated target — see `own_previous_architecture_diagram`.
    previous = await latest_document_content(db, request.tenant, request.source_component)
    previous_diagram = own_previous_architecture_diagram(previous)

    messages = build_adr_generation_prompt(transcript_text, qa_pairs, previous_diagram)
    response = await llm_complete(messages)
    document = response.choices[0].message.content

    # build_critic_prompt expects the richer ClarificationItem shape (keyed by `target`) —
    # SilverClarification's own audit trail only ever stored question/answer, so `target` is
    # filled with the source name here just to satisfy that shape; the Critic only uses it for
    # display, never to resolve anything.
    clarification_items = [
        {"target": request.source_component, "question": qa["question"], "answer": qa["answer"]}
        for qa in qa_pairs
    ]
    critic_messages = build_critic_prompt(document, transcript_text, clarification_items)
    critic_response = await llm_complete(critic_messages, response_format=CritiqueResult)
    critique = CritiqueResult.model_validate(load_json_response(critic_response.choices[0].message.content))

    return RegenerateResponse(
        document=document, score=critique.completeness_score, unresolved_points=critique.unresolved_points
    )


class AskMoreRequest(BaseModel):
    tenant: str
    source_component: str
    max_questions_per_stage: int


class AskMoreResponse(BaseModel):
    questions: list[str]


@router.post("/transcriptions/ask-more", response_model=AskMoreResponse)
async def ask_more_questions(request: AskMoreRequest, db: AsyncSession = Depends(get_session)) -> AskMoreResponse:
    """Drafts one fresh round of clarification questions for a draft that already went through
    its first pass — the same two-stage generation `generate_architecture_questions`/
    `generate_data_contract_questions` run inside the graph, plus the same classification
    `classify_questions` runs, called standalone here: no Boss, no persistence, same reasoning
    as `regenerate_document`. Lets the reviewer pull for more detail with targeted questions
    instead of writing free-text feedback by hand — the frontend answers them and calls
    `/transcriptions/regenerate` with the Q&A folded into one feedback string, reusing that
    endpoint rather than duplicating its Actor/Critic call.

    Grounds the new questions on this draft's own latest content via
    `previous_architecture_context` (its §3/§4 — everything already established, including this
    session's own earlier answers, not its §2 — see `own_previous_architecture_diagram`'s
    docstring for why that distinction matters elsewhere) so it asks about what is still
    missing instead of re-asking anything already resolved.

    `_top_questions` is `agents.graph`'s own private dedupe-and-cap helper, reused here (not
    reimplemented) for the same reason `_persist_document_version` is reused by
    `finalize_document` below — one algorithm, not a second copy that could drift from it."""
    bind_tenant(request.tenant)
    transcript_text = await bronze_content_for_source(db, request.tenant, request.source_component)
    if not transcript_text:
        raise HTTPException(
            status_code=404, detail=f"No bronze content found for {request.source_component!r}"
        )
    ingestion_date_str = await bronze_ingestion_date_for_source(
        db, request.tenant, request.source_component
    ) or date.today().strftime("%Y%m%d")
    bronze_rows = [{"source_component": request.source_component, "content": transcript_text}]

    previous = await latest_document_content(db, request.tenant, request.source_component)
    known_architecture = previous_architecture_context(previous)

    architecture_result = await generate_architecture_questions_for_batch(
        ingestion_date_str, bronze_rows, architecture_diagram=known_architecture
    )
    contract_result = await generate_data_contract_questions_for_batch(
        ingestion_date_str, bronze_rows, architecture_result.mentioned_data_contracts
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
    tenant: str
    source_component: str
    content: str


class FinalizeResponse(BaseModel):
    version: int


@router.post("/transcriptions/finalize", response_model=FinalizeResponse)
async def finalize_document(request: FinalizeRequest, db: AsyncSession = Depends(get_session)) -> FinalizeResponse:
    """"Publish": persists `content` — whichever draft the reviewer is looking at, original or
    regenerated — as this source's next SilverDocument version (a no-op version bump if it's
    byte-identical to what's already there), and runs Gold extraction on it. That's the whole
    action: publishing an ADR means its facts are in Gold and retrievable from Chat
    immediately, with no separate approval step. Reuses `_persist_document_version` (the same
    hash-compare-then-bump `write_document` uses inside the graph) and `gold_service.
    extract_and_persist_gold_facts` (the same shared body `scripts/backfill_gold.py` uses) —
    this is the second and third real callers of each, not a third/fourth copy of either."""
    bind_tenant(request.tenant)
    latest_row = (
        await db.execute(
            select(SilverDocument)
            .where(
                SilverDocument.tenant == request.tenant,
                SilverDocument.source_component == request.source_component,
            )
            .order_by(SilverDocument.version.desc())
            .limit(1)
        )
    ).scalars().first()
    if latest_row is None:
        raise HTTPException(
            status_code=404,
            detail=f"No existing SilverDocument for {request.source_component!r} — upload it first",
        )
    ingestion_date = latest_row.ingestion_date

    version = await _persist_document_version(
        db, ingestion_date, request.source_component, request.content, [], [], request.tenant
    )
    await db.commit()

    await db.execute(
        delete(SilverChunk).where(
            SilverChunk.tenant == request.tenant,
            SilverChunk.source_component == request.source_component,
            SilverChunk.version == version,
        )
    )
    [embedding] = await asyncio.to_thread(embed, [request.content])
    db.add(
        SilverChunk(
            tenant=request.tenant,
            ingestion_date=ingestion_date,
            source_component=request.source_component,
            version=version,
            content=request.content,
            embedding=embedding,
        )
    )
    await db.commit()

    await gold_service.extract_and_persist_gold_facts(
        db, request.content, request.source_component, version, ingestion_date, tenant=request.tenant
    )
    await db.commit()

    return FinalizeResponse(version=version)


# --- Architecture history ------------------------------------------------------------------


class ArchitectureHistoryAdr(BaseModel):
    source_component: str
    version: int
    ingestion_date: str
    content: str


class ArchitectureHistoryResponse(BaseModel):
    diagram: str
    adrs: list[ArchitectureHistoryAdr]


@router.get("/architecture-history", response_model=ArchitectureHistoryResponse)
async def architecture_history(
    tenant: str, db: AsyncSession = Depends(get_session)
) -> ArchitectureHistoryResponse:
    """Backs the "Architecture history" tab: a read-only view of every ADR this tenant has on
    record (newest first), plus a live Mermaid diagram of Gold's current component state
    (`gold_service.current_architecture_diagram`, not any single ADR's own LLM-drawn diagram)."""
    bind_tenant(tenant)
    docs = (
        await db.execute(
            select(SilverDocument).where(SilverDocument.tenant == tenant).order_by(SilverDocument.id.desc())
        )
    ).scalars().all()
    diagram = await gold_service.current_architecture_diagram(db, tenant=tenant)
    return ArchitectureHistoryResponse(
        diagram=diagram,
        adrs=[
            ArchitectureHistoryAdr(
                source_component=doc.source_component,
                version=doc.version,
                ingestion_date=doc.ingestion_date.isoformat(),
                content=doc.content,
            )
            for doc in docs
        ],
    )


# --- Chat with RAG -------------------------------------------------------------------------


class ChatRequest(BaseModel):
    tenant: str
    question: str
    k: int = 8


class ChatResponse(BaseModel):
    answer: str
    retrieved: list[dict]


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, db: AsyncSession = Depends(get_session)) -> ChatResponse:
    """Answers from Gold — every fact `Publish` has ever persisted for this tenant is
    retrievable immediately, with no separate approval gate."""
    bind_tenant(request.tenant)
    vector = await gold_service.embed_question(request.question)
    rows = await gold_service.top_k_gold_evolution(db, vector, k=request.k, tenant=request.tenant)

    if not rows:
        return ChatResponse(answer="No Gold facts were relevant to this question.", retrieved=[])

    latest = await gold_service.latest_versions(db, rows)
    answer = await gold_service.answer_question(request.question, rows, latest)
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


class TestMonitorResponse(BaseModel):
    suites: list[dict]
    llm_usage_by_tenant: dict[str, dict]


@router.get("/test-monitor", response_model=TestMonitorResponse)
async def test_monitor() -> TestMonitorResponse:
    """"For all application", per the tab's own description — this is a global, app-wide view
    (every tenant's spend, every test suite), not scoped to the caller's own tenant, and gated
    only by `start_test_mode` at the frontend (this endpoint itself stays reachable; the tab
    that calls it is what `start_test_mode` hides)."""
    if not settings.start_test_mode:
        raise HTTPException(status_code=404, detail="Test monitor is disabled (start_test_mode=False)")

    suites = []
    for result_path in sorted(Path(".").glob("testing_*/output/result.json")):
        suites.append({"suite": result_path.parent.parent.name, "results": json.loads(result_path.read_text())})

    usage_by_tenant: dict[str, dict] = {}
    if LLM_USAGE_LOG.exists():
        for line in LLM_USAGE_LOG.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            bucket = usage_by_tenant.setdefault(
                entry.get("tenant", "default"),
                {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_eur": 0.0},
            )
            bucket["calls"] += 1
            bucket["input_tokens"] += entry.get("input_tokens") or 0
            bucket["output_tokens"] += entry.get("output_tokens") or 0
            bucket["cost_eur"] += entry.get("cost_eur") or 0.0

    return TestMonitorResponse(suites=suites, llm_usage_by_tenant=usage_by_tenant)
