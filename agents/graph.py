"""The Silver clarification loop, end-to-end — `doc/silver_process.md` §3 turned into code.

One deliberately large module instead of one file per node: the ten nodes below only
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
import hashlib
from pathlib import Path
from typing import Literal

import logfire
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt
from sqlalchemy import delete, select

from agents.prompts import build_adr_generation_prompt, build_classification_prompt, build_critic_prompt
from agents.schemas import ClassificationResult, CritiqueResult
from agents.service import (
    NoBronzeDocumentsError,
    distinct_sources,
    generate_architecture_questions_for_batch,
    generate_data_contract_questions_for_batch,
    load_bronze_rows,
    source_content,
    transcription_base_name,
)
from agents.state import ClarificationItem, SilverState
from agents.template import load_json_response
from app.config import settings
from db.models import SilverChunk, SilverClarification, SilverDocument
from db.session import async_session_factory
from ingestion.bronze_documents_chunker import parse_ingestion_date
from ingestion.embedder import embed
from llm import router

logfire.configure(
    token=settings.logfire_token,
    send_to_logfire="if-token-present",
    service_name="silver-clarification-loop",
)

_DECLINE_PHRASES = {"", "no sé", "no se", "unknown", "n/a", "idk", "i don't know"}
_DOWNGRADE_MARKER = " **[unknown — flagged by review]**"

_QUESTIONS_PRIORITIES = {
    "adr": 0,      # Decisions
    "component": 1,
    "architecture": 2,
    "change_impact": 3,
    "migration": 4,
    "metadata": 5,
}


def _top_questions(questions: list[str], scopes: dict[str, str] | None = None) -> list[str]:
    """When `scopes` (question text -> scope) is given — classify-origin questions, which carry
    a real scope — splits into the architecture bucket (everything except `data_contract`,
    ranked by `_QUESTIONS_PRIORITIES`, capped at `settings.max_architecture_pending_questions`) and
    the data-contract bucket (capped at `settings.max_data_contract_pending_questions`,
    truncated in drafted order). Boss-origin escalations (see `boss_decide`) have no scope of
    their own (synthesized ad hoc from a Critic claim) and are already naturally few, so
    without `scopes` they're just truncated to the architecture cap in their existing order."""
    if scopes is None:
        return questions[: settings.max_architecture_pending_questions]

    architecture_qs = [q for q in questions if scopes.get(q) != "data_contract"]
    data_contract_qs = [q for q in questions if scopes.get(q) == "data_contract"]
    architecture_qs.sort(key=lambda q: _QUESTIONS_PRIORITIES.get(scopes.get(q, ""), len(_QUESTIONS_PRIORITIES)))

    return (
        architecture_qs[: settings.max_architecture_pending_questions]
        + data_contract_qs[: settings.max_data_contract_pending_questions]
    )


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


async def load_bronze(state: SilverState) -> dict:
    async with async_session_factory() as session:
        rows = await load_bronze_rows(state["ingestion_date"], session)

    return {
        "bronze_documents": rows,
        "transcript_text": " ".join(r["content"] for r in rows),
    }


async def generate_architecture_questions(state: SilverState) -> dict:
    """Architecture questions and mentions"""
    result = await generate_architecture_questions_for_batch(
        state["ingestion_date"], state["bronze_documents"], architecture_diagram=""
    )
    return {
        "generated_questions": [q.model_dump() for q in result.questions],
        "mentioned_components": [c.model_dump() for c in result.mentioned_components],
        "mentioned_data_contracts": [c.model_dump() for c in result.mentioned_data_contracts],
    }


async def generate_data_contract_questions(state: SilverState) -> dict:
    """Data contracts questions and mentions"""
    result = await generate_data_contract_questions_for_batch(
        state["ingestion_date"], state["bronze_documents"], state["mentioned_data_contracts"]
    )
    return {
        "generated_questions": state["generated_questions"] + [q.model_dump() for q in result.questions],
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
    needs_clarification = [c for c in clarifications if c["status"] == "needs_clarification"]
    pending = _top_questions(
        [c["question"] for c in needs_clarification],
        scopes={c["question"]: c["scope"] for c in needs_clarification},
    )

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
    """The Actor. One LLM call per distinct source, writing the final ADR directly —
    `prompts/adr_generator.jinja`'s own structure (`doc/silver_process.md`
    §3 node 6). Takes a flat question/answer list, not the graph's own richer
    `ClarificationItem` shape — see `agents.prompts.QaPair` — so
    `id`/`scope`/`target`/`requirement` are dropped here; the ADR prompt only ever reads
    the question text and its answer."""
    sources = state["redraft_only"] or distinct_sources(state["bronze_documents"])
    qa_pairs = [{"question": c["question"], "answer": c["answer"]} for c in state["clarifications"]]

    documents = dict(state["documents"])
    for source in sources:
        messages = build_adr_generation_prompt(source_content(state["bronze_documents"], source), qa_pairs)
        response = await router.complete(messages)
        documents[source] = response.choices[0].message.content

    return {
        "documents": documents,
        "active_sources": sources,
        "redraft_only": None,
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
        update["pending_questions"] = _top_questions(pending_questions)
        update["interrupt_origin"] = "boss"
    return update


def route_after_boss(state: SilverState) -> Literal["ask_human", "write_document"]:
    return "ask_human" if state["pending_questions"] else "write_document"


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


async def _persist_document_version(
    session, ingestion_date, source_component: str, content: str
) -> int:
    """Writes one `SilverDocument` row, deciding the version deterministically from
    content alone — never from what the LLM says about itself. Compares the new ADR's
    hash against the latest existing row for this `source_component`: identical hash
    overwrites that same row in place (idempotent re-run, no new version); a different
    hash inserts a new row at `latest_version + 1`, leaving the older version's row
    untouched — both stay queryable. Returns the version actually written, so
    `chunk_and_embed` doesn't have to re-derive it."""
    new_hash = _content_hash(content)
    latest = (
        await session.execute(
            select(SilverDocument)
            .where(SilverDocument.source_component == source_component)
            .order_by(SilverDocument.version.desc())
            .limit(1)
        )
    ).scalar_one_or_none()

    if latest is None:
        version = 1
        session.add(
            SilverDocument(
                ingestion_date=ingestion_date,
                source_component=source_component,
                version=version,
                content=content,
                content_hash=new_hash,
            )
        )
    elif latest.content_hash == new_hash:
        version = latest.version
        latest.ingestion_date = ingestion_date
        latest.content = content
    else:
        version = latest.version + 1
        session.add(
            SilverDocument(
                ingestion_date=ingestion_date,
                source_component=source_component,
                version=version,
                content=content,
                content_hash=new_hash,
            )
        )
    return version


def _write_adr_audit_file(ingestion_date_str: str, source_component: str, content: str) -> None:
    """Writes `output/ingestion_date=<date>/adr/<transcription>.md` — a human-inspectable
    audit copy of the ADR `write_document` just persisted, same convention
    `agents.service._write_json_audit_file` already uses for the two question-generation
    stages' own audit files (`doc/silver_process.md` §1). Always reflects this run's latest
    content, unversioned on disk — only `silver_documents` keeps version history; this
    file exists to be read by a human, not by anything downstream."""
    adr_dir = Path(settings.output_dir) / f"ingestion_date={ingestion_date_str}" / "adr"
    adr_dir.mkdir(parents=True, exist_ok=True)
    (adr_dir / f"{transcription_base_name(source_component)}.md").write_text(content, encoding="utf-8")


async def write_document(state: SilverState) -> dict:
    """Persists each synthesized ADR at its own (`source_component`, `version`) —
    see `_persist_document_version` for the overwrite-vs-new-version decision — and
    appends this run's clarifications to `silver_clarifications` — one row per
    (source, question), inserted fresh every run rather than upserted, so it
    accumulates a history instead of only the latest state. Also writes each ADR to disk
    (`_write_adr_audit_file`) for manual inspection alongside the Postgres write."""
    ingestion_date = parse_ingestion_date(state["ingestion_date"])
    document_versions: dict[str, int] = dict(state["document_versions"])
    async with async_session_factory() as session:
        for source, content in state["documents"].items():
            document_versions[source] = await _persist_document_version(
                session, ingestion_date, source, content
            )
            _write_adr_audit_file(state["ingestion_date"], source, content)

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

    return {"document_versions": document_versions}


async def chunk_and_embed(state: SilverState) -> dict:
    """One embedding per ADR — no token-splitting: an ADR is retrieved whole, never as
    a fragment (`doc/silver_process.md` §3 node 10). Upserts exactly the
    (`source_component`, `version`) pair `write_document` just wrote, leaving every
    other version's chunk row (older versions, other sources, other dates) untouched —
    the opposite of the old blanket delete-by-`ingestion_date`, which would have wiped
    out the version history this node now exists to keep. `END`."""
    ingestion_date = parse_ingestion_date(state["ingestion_date"])
    async with async_session_factory() as session:
        for source, content in state["documents"].items():
            if not content.strip():
                continue
            version = state["document_versions"][source]
            await session.execute(
                delete(SilverChunk).where(
                    SilverChunk.source_component == source, SilverChunk.version == version
                )
            )
            [embedding] = await asyncio.to_thread(embed, [content])
            session.add(
                SilverChunk(
                    ingestion_date=ingestion_date,
                    source_component=source,
                    version=version,
                    content=content,
                    embedding=embedding,
                )
            )
        await session.commit()
    return {}


# --- Graph assembly ------------------------------------------------------------


def build_graph(checkpointer) -> CompiledStateGraph:
    graph = StateGraph(SilverState)

    graph.add_node("load_bronze", load_bronze)
    graph.add_node("generate_architecture_questions", generate_architecture_questions)
    graph.add_node("generate_data_contract_questions", generate_data_contract_questions)
    graph.add_node("classify_questions", classify_questions)
    graph.add_node("ask_human", ask_human)
    graph.add_node("synthesize_document", synthesize_document)
    graph.add_node("critic_document", critic_document)
    graph.add_node("boss_decide", boss_decide)
    graph.add_node("write_document", write_document)
    graph.add_node("chunk_and_embed", chunk_and_embed)

    graph.add_edge(START, "load_bronze")
    graph.add_edge("load_bronze", "generate_architecture_questions")
    graph.add_edge("generate_architecture_questions", "generate_data_contract_questions")
    graph.add_edge("generate_data_contract_questions", "classify_questions")
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
