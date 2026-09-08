"""The Silver clarification loop, end-to-end — `doc/silver_process.md` §3 turned into code.

One deliberately large module instead of one file per node: the nine nodes below only
make sense wired together (see `.tmp/tasks.md` Task 6 for why splitting the commit
would mean landing broken intermediate states).
"""

import os

# Must run before `langgraph` (imported below) is loaded anywhere in the process —
# LangGraph ships its own OpenTelemetry tracing via the bundled LangSmith SDK, and it
# reads these at import time. Once `logfire.configure()` below installs the global OTel
# tracer provider, LangSmith's tracer detects it and every graph-run/node span flows
# straight into Logfire — no separate LangSmith account, exporter, or API key needed.
os.environ.setdefault("LANGSMITH_OTEL_ENABLED", "true")
os.environ.setdefault("LANGSMITH_OTEL_ONLY", "true")
os.environ.setdefault("LANGSMITH_TRACING", "true")

import asyncio
from typing import Literal

import logfire
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt
from sqlalchemy import delete, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import ProgrammingError

from agents.prompts import build_classification_prompt, build_critic_prompt, build_synthesis_prompt
from agents.schemas import ClassificationResult, CritiqueResult
from agents.service import (
    NoBronzeDocumentsError,
    distinct_sources,
    generate_questions_for_batch,
    load_bronze_rows,
    source_content,
)
from agents.state import ClarificationItem, GoldComponentSnapshot, SilverState
from agents.template import load_json_response, load_template
from app.config import settings
from db.models import SilverChunk, SilverClarification, SilverDocument
from db.session import async_session_factory
from ingestion.bronze_documents_chunker import chunk_text, parse_ingestion_date
from ingestion.embedder import embed
from llm import router

logfire.configure(
    token=settings.logfire_token,
    send_to_logfire="if-token-present",
    service_name="silver-clarification-loop",
)

_DECLINE_PHRASES = {"", "no sé", "no se", "unknown", "n/a", "idk", "i don't know"}
_DOWNGRADE_MARKER = " **[unknown — flagged by review]**"


def checkpointer_dsn() -> str:
    """`AsyncPostgresSaver` speaks psycopg's DSN format, not SQLAlchemy's `+asyncpg`
    one — same `technical_meeting_rag` database either way, no new infra."""
    return settings.database_url.replace("postgresql+asyncpg://", "postgresql://")


def _downgrade_claim(content: str, claim: str) -> str:
    if not claim or claim not in content:
        return content
    marked = claim + _DOWNGRADE_MARKER
    if marked in content:
        return content  # already downgraded — idempotent against a second pass
    return content.replace(claim, marked, 1)


async def _lookup_known_gold_components() -> list[GoldComponentSnapshot]:
    """Best-effort snapshot of Gold's `components` table. Tolerates the table not
    existing yet — Task 7 (Gold's minimal schema) may not have landed, and Silver
    must work either way (`doc/silver_process.md` §7)."""
    async with async_session_factory() as session:
        try:
            result = await session.execute(text("SELECT name, description, profile FROM components"))
        except ProgrammingError:
            await session.rollback()
            return []
        return [dict(row) for row in result.mappings().all()]


# --- Nodes -------------------------------------------------------------------
# load_bronze and generate_questions are thin wrappers — their actual logic lives in
# agents/service.py, callable directly (make questions, unit tests) without a graph.


async def load_bronze(state: SilverState) -> dict:
    async with async_session_factory() as session:
        rows = await load_bronze_rows(state["ingestion_date"], session)

    return {
        "bronze_documents": rows,
        "transcript_text": " ".join(r["content"] for r in rows),
    }


async def generate_questions(state: SilverState) -> dict:
    """See `agents.service.generate_questions_for_batch` for what this actually does
    (`doc/silver_process.md` §3 node 2) — this node just supplies the graph's state."""
    result = await generate_questions_for_batch(
        state["ingestion_date"], state["bronze_documents"], architecture_diagram=""
    )
    return {
        "generated_questions": [q.model_dump() for q in result.questions],
        "mentioned_components": [c.model_dump() for c in result.mentioned_components],
    }


async def classify_questions(state: SilverState) -> dict:
    questions = state["generated_questions"]
    messages = build_classification_prompt(questions, state["transcript_text"])
    response = await router.complete(messages, response_format=ClassificationResult)
    result = ClassificationResult.model_validate(load_json_response(response.choices[0].message.content))

    # Matched back to the drafted question by `id`, not by re-sent text — an id the
    # classifier hallucinated (not among the questions it was given) is dropped rather
    # than crashing the node on a malformed response.
    by_id = {q["id"]: q for q in questions}
    clarifications: list[ClarificationItem] = []
    for c in result.classifications:
        question = by_id.get(c.id)
        if question is None:
            continue
        clarifications.append(
            {
                "id": question["id"],
                "scope": question["scope"],
                "target": question["target"],
                "requirement": question["requirement"],
                "question": question["question"],
                "answer": c.answer,
                "status": c.status,
            }
        )
    pending = [c["question"] for c in clarifications if c["status"] == "needs_clarification"]

    return {
        "clarifications": clarifications,
        "pending_questions": pending,
        "interrupt_origin": "classify",
    }


def route_after_classify(state: SilverState) -> Literal["ask_human", "synthesize_document"]:
    return "ask_human" if state["pending_questions"] else "synthesize_document"


def ask_human(state: SilverState) -> dict:
    """One `interrupt()` carrying every pending question together — reused for both
    the classify-stage gap-filling (`interrupt_origin == "classify"`) and a Boss
    escalation over a contradiction (`interrupt_origin == "boss"`); same mechanism,
    same Postgres-backed checkpointer, no new pause path for the second case."""
    payload = {
        "ingestion_date": state["ingestion_date"],
        "pending_questions": state["pending_questions"],
        "origin": state["interrupt_origin"],
    }
    human_answers: dict[str, str] = interrupt(payload)

    updated = list(state["clarifications"])
    existing_questions = {item["question"] for item in updated}

    for question in state["pending_questions"]:
        raw_answer = human_answers.get(question, "")
        declined = raw_answer.strip().lower() in _DECLINE_PHRASES
        answer = None if declined else raw_answer.strip()
        status: Literal["unknown", "human_answered"] = "unknown" if declined else "human_answered"

        if question in existing_questions:
            for item in updated:
                if item["question"] == question:
                    item["answer"] = answer
                    item["status"] = status
        else:
            # A Boss-origin escalation question (see boss_decide) — synthesized on the
            # spot from a Critic claim, so it never went through generate_questions and
            # has no real id/scope/target of its own. Fill in placeholders that still
            # identify what it's about, so this item's shape matches every other one.
            updated.append(
                {
                    "id": f"boss-escalation:{len(updated)}",
                    "scope": "change_impact",
                    "target": state["ingestion_date"],
                    "requirement": "contradiction_resolution",
                    "question": question,
                    "answer": answer,
                    "status": status,
                }
            )

    return {"clarifications": updated, "pending_questions": []}


async def synthesize_document(state: SilverState) -> dict:
    """The Actor. One LLM call per distinct source, writing directly into
    `doc/clarification_template.md`'s structure (`doc/silver_process.md` §3 node 5)."""
    template_text = load_template()
    sources = state["redraft_only"] or distinct_sources(state["bronze_documents"])
    known_gold_components = state["known_gold_components"] or await _lookup_known_gold_components()

    documents = dict(state["documents"])
    for source in sources:
        messages = build_synthesis_prompt(
            template_text,
            source_content(state["bronze_documents"], source),
            state["clarifications"],
            known_gold_components,
            state["mentioned_components"],
        )
        response = await router.complete(messages)
        documents[source] = response.choices[0].message.content

    return {
        "documents": documents,
        "active_sources": sources,
        "redraft_only": None,
        "known_gold_components": known_gold_components,
    }


async def critic_document(state: SilverState) -> dict:
    """Mandatory, always runs, no confidence threshold that skips it."""
    critiques = dict(state["critiques"])
    for source in state["active_sources"]:
        messages = build_critic_prompt(
            state["documents"][source],
            source_content(state["bronze_documents"], source),
            state["clarifications"],
        )
        response = await router.complete(messages, response_format=CritiqueResult)
        result = CritiqueResult.model_validate(load_json_response(response.choices[0].message.content))
        critiques[source] = [c.model_dump() for c in result.claims]

    return {"critiques": critiques}


def boss_decide(state: SilverState) -> dict:
    """Deterministic, no LLM call — a policy over the Critic's output, not a second
    opinion. Escalates a real contradiction once; downgrades everything else."""
    documents = dict(state["documents"])
    boss_verdicts = dict(state["boss_verdicts"])
    revision_attempted = dict(state["revision_attempted"])
    escalate_sources: list[str] = []
    pending_questions: list[str] = []

    for source in state["active_sources"]:
        claims = state["critiques"].get(source, [])
        material = [c for c in claims if c["severity"] == "material"]
        low = [c for c in claims if c["severity"] == "low"]

        content = documents[source]
        for claim in low:
            content = _downgrade_claim(content, claim["claim"])

        if not material:
            documents[source] = content
            boss_verdicts[source] = "ok"
            continue

        if not revision_attempted.get(source, False):
            documents[source] = content
            boss_verdicts[source] = "needs_human_review"
            revision_attempted[source] = True
            escalate_sources.append(source)
            for claim in material:
                pending_questions.append(
                    f'[{source}] The transcript does not support this claim: "{claim["claim"]}" '
                    f'— {claim["rationale"]}. What is actually correct?'
                )
        else:
            for claim in material:
                content = _downgrade_claim(content, claim["claim"])
            documents[source] = content
            boss_verdicts[source] = "ok"

    update: dict = {
        "documents": documents,
        "boss_verdicts": boss_verdicts,
        "revision_attempted": revision_attempted,
    }
    if escalate_sources:
        update["redraft_only"] = escalate_sources
        update["pending_questions"] = pending_questions
        update["interrupt_origin"] = "boss"
    return update


def route_after_boss(state: SilverState) -> Literal["ask_human", "write_document"]:
    return "ask_human" if state["pending_questions"] else "write_document"


async def write_document(state: SilverState) -> dict:
    """Upserts `silver_documents` (deterministic, no LLM call — overwrite-in-place
    idempotency, same pattern `scripts/download_transcript.py` used for re-downloads
    before it moved to timestamped snapshots) and appends this run's clarifications to
    `silver_clarifications` — one row per (source, question), inserted fresh every run
    rather than upserted, so it accumulates a history instead of only the latest state."""
    ingestion_date = parse_ingestion_date(state["ingestion_date"])
    async with async_session_factory() as session:
        for source, content in state["documents"].items():
            stmt = pg_insert(SilverDocument).values(
                ingestion_date=ingestion_date,
                source_component=source,
                content=content,
            )
            stmt = stmt.on_conflict_do_update(
                index_elements=[SilverDocument.source_component],
                set_={
                    "ingestion_date": stmt.excluded.ingestion_date,
                    "content": stmt.excluded.content,
                },
            )
            await session.execute(stmt)

            for item in state["clarifications"]:
                session.add(
                    SilverClarification(
                        ingestion_date=ingestion_date,
                        source_component=source,
                        question=item["question"],
                        answer=item["answer"],
                    )
                )
        await session.commit()

    return {}


async def chunk_and_embed(state: SilverState) -> dict:
    """Deletes any existing `silver_chunks` for this date (re-run safety), reuses
    Bronze's own chunker and embedder unchanged. `END`."""
    ingestion_date = parse_ingestion_date(state["ingestion_date"])
    async with async_session_factory() as session:
        await session.execute(delete(SilverChunk).where(SilverChunk.ingestion_date == ingestion_date))

        for source, content in state["documents"].items():
            chunks = chunk_text(content, settings.chunk_size_tokens, settings.chunk_overlap_tokens)
            if not chunks:
                continue
            embeddings = await asyncio.to_thread(embed, chunks)
            for chunk_content, embedding in zip(chunks, embeddings, strict=True):
                session.add(
                    SilverChunk(
                        ingestion_date=ingestion_date,
                        source_component=source,
                        content=chunk_content,
                        embedding=embedding,
                    )
                )
        await session.commit()
    return {}


# --- Graph assembly ------------------------------------------------------------


def build_graph(checkpointer) -> CompiledStateGraph:
    graph = StateGraph(SilverState)

    graph.add_node("load_bronze", load_bronze)
    graph.add_node("generate_questions", generate_questions)
    graph.add_node("classify_questions", classify_questions)
    graph.add_node("ask_human", ask_human)
    graph.add_node("synthesize_document", synthesize_document)
    graph.add_node("critic_document", critic_document)
    graph.add_node("boss_decide", boss_decide)
    graph.add_node("write_document", write_document)
    graph.add_node("chunk_and_embed", chunk_and_embed)

    graph.add_edge(START, "load_bronze")
    graph.add_edge("load_bronze", "generate_questions")
    graph.add_edge("generate_questions", "classify_questions")
    graph.add_conditional_edges(
        "classify_questions",
        route_after_classify,
        {"ask_human": "ask_human", "synthesize_document": "synthesize_document"},
    )
    # ask_human always continues to synthesize_document regardless of origin — a
    # classify-origin resume has just resolved every pending question (nothing left
    # to loop back to route_after_classify for); a boss-origin resume redrafts the
    # flagged source only (`redraft_only`, set by boss_decide before interrupting).
    graph.add_edge("ask_human", "synthesize_document")
    graph.add_edge("synthesize_document", "critic_document")
    graph.add_edge("critic_document", "boss_decide")
    graph.add_conditional_edges(
        "boss_decide",
        route_after_boss,
        {"ask_human": "ask_human", "write_document": "write_document"},
    )
    graph.add_edge("write_document", "chunk_and_embed")
    graph.add_edge("chunk_and_embed", END)

    return graph.compile(checkpointer=checkpointer)
