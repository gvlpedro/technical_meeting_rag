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
from pathlib import Path
from typing import Literal

import logfire
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt
from sqlalchemy import delete, select

from agents import gold_service
from agents.prompts import build_adr_generation_prompt, build_classification_prompt, build_critic_prompt
from agents.schemas import (
    ArchitecturePayload,
    ClassificationResult,
    ComponentPayload,
    CritiqueResult,
    DataContractPayload,
)
from agents.service import (
    NoBronzeDocumentsError,
    distinct_sources,
    extract_authors_line,
    generate_architecture_questions_for_batch,
    generate_data_contract_questions_for_batch,
    insert_authors_line,
    latest_document_content,
    load_bronze_rows,
    mentions_grounded_in_source,
    previous_architecture_context,
    previous_target_architecture_diagram,
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

_DECLINE_PHRASES = {"", "no sé", "no se", "unknown", "n/a", "idk", "i don't know", "[irrelevant]"}
_DOWNGRADE_MARKER = " **[unknown — flagged by review]**"

# The frontend's per-question "Infer an answer"/"Suggest info" buttons (see `frontend/app.py`)
# send one of these two literal strings as the answer instead of free text — never added to
# `_DECLINE_PHRASES` above, since they must survive into `SilverClarification.answer` as real
# answered text for `prompts/adr_generator.jinja`'s own CLARIFICATION ANSWER MARKERS section to
# interpret. "Irrelevant" (unlike these two) genuinely means "no answer" and belongs in the
# decline set instead — it has no special downstream interpretation of its own.
INFER_FROM_CONTEXT_MARKER = "[INFER FROM CONTEXT]"
SUGGEST_INFO_MARKER = "[SUGGEST INFO]"

_QUESTIONS_PRIORITIES = {
    "adr": 0,      # Decisions
    "component": 1,
    "architecture": 2,
    "change_impact": 3,
    "migration": 4,
    "metadata": 5,
}


def _top_questions(
    questions: list[str],
    architecture_cap: int,
    data_contract_cap: int,
    scopes: dict[str, str] | None = None,
) -> list[str]:
    """When `scopes` (question text -> scope) is given — classify-origin questions, which carry
    a real scope — splits into the architecture bucket (everything except `data_contract`,
    ranked by `_QUESTIONS_PRIORITIES`, capped at `architecture_cap`) and the data-contract
    bucket (capped at `data_contract_cap`, truncated in drafted order). Boss-origin escalations
    (see `boss_decide`) have no scope of their own (synthesized ad hoc from a Critic claim) and
    are already naturally few, so without `scopes` they're just truncated to `architecture_cap`
    in their existing order.

    Both caps come from the run's own state (`state["max_architecture_pending_questions"]`/
    `state["max_data_contract_pending_questions"]`), not read from `settings` directly here —
    see `agents.state.initial_state` for where a caller (the frontend's upload form, a script,
    a test) can override them, falling back to `settings`'s own defaults otherwise.

    De-duplicates by exact text first, preserving first-seen order — two identical question
    strings (`boss_decide` quoting the same Critic claim text twice, or the classify stage
    drafting the same wording twice) must never reach `ask_human` as two separate pending
    questions: this node's own resume payload and the frontend's one-widget-per-question
    rendering both key off question text, so a real duplicate either silently collapses to one
    answer or crashes the frontend outright (Streamlit requires unique widget keys)."""
    questions = list(dict.fromkeys(questions))
    if scopes is None:
        return questions[:architecture_cap]

    architecture_qs = [q for q in questions if scopes.get(q) != "data_contract"]
    data_contract_qs = [q for q in questions if scopes.get(q) == "data_contract"]
    architecture_qs.sort(key=lambda q: _QUESTIONS_PRIORITIES.get(scopes.get(q, ""), len(_QUESTIONS_PRIORITIES)))

    return architecture_qs[:architecture_cap] + data_contract_qs[:data_contract_cap]


def checkpointer_dsn() -> str:
    """`AsyncPostgresSaver` speaks psycopg's DSN format, not SQLAlchemy's `+asyncpg`
    one — same `technical_meeting_rag` database either way, no new infra."""
    return settings.database_url.replace("postgresql+asyncpg://", "postgresql://")


def _is_inside_code_fence(content: str, index: int) -> bool:
    """True if `index` falls inside a ` ``` `-fenced block — an odd number of fence markers
    before it means we're currently between an opening and a closing one."""
    return content.count("```", 0, index) % 2 == 1


def _downgrade_claim(content: str, claim: str) -> str:
    """Appends `_DOWNGRADE_MARKER` right after the first occurrence of `claim` that is NOT
    inside a ` ```mermaid ` fence — the ADR's own "previous"/"target architecture" diagrams
    live in fenced blocks (`prompts/adr_generator.jinja`'s OUTPUT STRUCTURE), and the
    Critic's claim text is a verbatim quote from anywhere in the document, diagram node
    labels included (`prompts/adr_critic.jinja`'s own quoting rule doesn't exempt them).
    Splicing `**[unknown — flagged by review]**` into a Mermaid label breaks its syntax —
    Mermaid has no idea what to do with a stray `**` — and the rendered diagram fails outright
    instead of just looking annotated. If `claim` only ever occurs inside a fence, this
    returns `content` unchanged rather than corrupt the diagram: there is no prose sentence
    to annotate, but the diagram must still render."""
    if not claim:
        return content
    marked = claim + _DOWNGRADE_MARKER
    if marked in content:
        return content  # already downgraded — idempotent against a second pass

    search_from = 0
    while True:
        index = content.find(claim, search_from)
        if index == -1:
            return content
        if not _is_inside_code_fence(content, index):
            return content[:index] + marked + content[index + len(claim) :]
        search_from = index + 1


async def load_bronze(state: SilverState) -> dict:
    async with async_session_factory() as session:
        rows = await load_bronze_rows(
            state["ingestion_date"],
            session,
            tenant=state["tenant"],
            source_components=state["source_components"],
        )

    return {
        "bronze_documents": rows,
        "transcript_text": " ".join(r["content"] for r in rows),
    }


async def generate_architecture_questions(state: SilverState) -> dict:
    """Architecture questions and mentions.

    `architecture_diagram` (this stage's own KNOWN_ARCHITECTURE input) is built from each
    distinct source's own previous ADR, if one exists — Silver's own version history, never
    Gold (see `agents.service.previous_architecture_context`). A source with no prior ADR
    contributes nothing here, which is exactly "no prior architecture known" to the prompt."""
    known_architecture_parts = []
    async with async_session_factory() as session:
        for source in distinct_sources(state["bronze_documents"]):
            previous = await latest_document_content(session, state["tenant"], source)
            context = previous_architecture_context(previous)
            if context:
                known_architecture_parts.append(f"### {source}\n\n{context}")
    known_architecture = "\n\n".join(known_architecture_parts)

    result = await generate_architecture_questions_for_batch(
        state["ingestion_date"], state["bronze_documents"], architecture_diagram=known_architecture
    )
    # A name the LLM put in BOTH lists is a data contract wearing a component's clothes, not two
    # distinct entities — "Topic"/"Queue" is a legal component type (architecture_questions.jinja
    # PHASE 3) and a data contract is often named after the topic/queue it's delivered over, so
    # the two lists can genuinely collide on the same name even with the prompt's own "don't
    # double-list" instruction (still just prompt-following, not enforced). Preferring the data
    # contract classification here — mechanically, not by asking the LLM to try harder — means
    # every downstream consumer (grounding checks, Gold extraction, the ADR itself) sees one
    # consistent classification instead of each re-deciding independently.
    contract_names = {c.name.lower() for c in result.mentioned_data_contracts}
    components = [c for c in result.mentioned_components if c.name.lower() not in contract_names]
    return {
        "generated_questions": [q.model_dump() for q in result.questions],
        "mentioned_components": [c.model_dump() for c in components],
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
        state["max_architecture_pending_questions"],
        state["max_data_contract_pending_questions"],
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
    async with async_session_factory() as session:
        for source in sources:
            previous = await latest_document_content(session, state["tenant"], source)
            previous_diagram = previous_target_architecture_diagram(previous)
            messages = build_adr_generation_prompt(
                source_content(state["bronze_documents"], source), qa_pairs, previous_diagram
            )
            response = await router.complete(messages)
            documents[source] = response.choices[0].message.content

    return {
        "documents": documents,
        "active_sources": sources,
        "redraft_only": None,
    }


async def critic_document(state: SilverState) -> dict:
    """Mandatory, always runs, no confidence threshold that skips it. Also produces this
    source's `completeness_score`/`unresolved_points` (see `prompts/adr_critic.jinja`'s
    COMPLETENESS SCORING section) from the same read — no second LLM call needed just to
    grade the document separately from reviewing its claims."""
    critiques = dict(state["critiques"])
    adr_scores = dict(state["adr_scores"])
    adr_unresolved_points = dict(state["adr_unresolved_points"])
    for source in state["active_sources"]:
        messages = build_critic_prompt(
            state["documents"][source],
            source_content(state["bronze_documents"], source),
            state["clarifications"],
        )
        response = await router.complete(messages, response_format=CritiqueResult)
        result = CritiqueResult.model_validate(load_json_response(response.choices[0].message.content))
        critiques[source] = [c.model_dump() for c in result.claims]
        adr_scores[source] = result.completeness_score
        adr_unresolved_points[source] = result.unresolved_points

    return {
        "critiques": critiques,
        "adr_scores": adr_scores,
        "adr_unresolved_points": adr_unresolved_points,
    }


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
        update["pending_questions"] = _top_questions(
            pending_questions,
            state["max_architecture_pending_questions"],
            state["max_data_contract_pending_questions"],
        )
        update["interrupt_origin"] = "boss"
    return update


def route_after_boss(state: SilverState) -> Literal["ask_human", "write_document"]:
    return "ask_human" if state["pending_questions"] else "write_document"


async def _persist_document_version(
    session,
    ingestion_date,
    source_component: str,
    content: str,
    mentioned_component_names: list[dict],
    mentioned_data_contract_names: list[dict],
    tenant: str,
) -> int:
    """Writes one `SilverDocument` row, deciding the version deterministically from
    content alone — never from what the LLM says about itself. Compares the new ADR's
    hash against the latest existing row for this `source_component`: identical hash
    overwrites that same row in place (idempotent re-run, no new version); a different
    hash inserts a new row at `latest_version + 1`, leaving the older version's row
    untouched — both stay queryable. Returns the version actually written, so
    `chunk_and_embed` doesn't have to re-derive it.

    `mentioned_component_names`/`mentioned_data_contract_names` are refreshed on every write,
    including the same-hash overwrite branch — a re-run can legitimately draft a different
    (or differently-grounded) mention list even when the synthesized ADR text itself hashes
    identical. `_row_fields` is the single place all three branches (insert-first, overwrite,
    insert-next-version) read shared column values from, so a future column addition is a
    one-line change here instead of a hand-edit repeated across three constructor calls.

    `authored_by` is read straight out of `content`'s own `**Authors:**` line
    (`agents.service.extract_authors_line`) — never a separate parameter — so this function has
    exactly one source of truth for who authored a version, matching whatever the document
    itself says, including on the overwrite branch (an identical-hash re-run still refreshes it,
    same reasoning as the mention lists above)."""
    new_hash = gold_service.content_hash(content)
    authored_by = extract_authors_line(content)
    latest = (
        await session.execute(
            select(SilverDocument)
            .where(SilverDocument.tenant == tenant, SilverDocument.source_component == source_component)
            .order_by(SilverDocument.version.desc())
            .limit(1)
        )
    ).scalar_one_or_none()

    def _row_fields(version: int) -> dict:
        return {
            "tenant": tenant,
            "ingestion_date": ingestion_date,
            "source_component": source_component,
            "version": version,
            "content": content,
            "content_hash": new_hash,
            "mentioned_component_names": mentioned_component_names,
            "mentioned_data_contract_names": mentioned_data_contract_names,
            "authored_by": authored_by,
        }

    if latest is None:
        version = 1
        session.add(SilverDocument(**_row_fields(version)))
    elif latest.content_hash == new_hash:
        version = latest.version
        for field, value in _row_fields(version).items():
            setattr(latest, field, value)
    else:
        version = latest.version + 1
        session.add(SilverDocument(**_row_fields(version)))
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
    """Persists each synthesized ADR at its own (`source_component`, `version`) — see
    `_persist_document_version` for the overwrite-vs-new-version decision — and writes each ADR
    to disk (`_write_adr_audit_file`) for manual inspection alongside the Postgres write. Only
    when `state["persist"]` is true: the frontend's initial upload runs with it false, so this
    ADR (and everything downstream — see `route_after_write_document`) stays a draft until a
    human explicitly clicks "Publish". Publishing later calls `_persist_document_version` itself
    (`finalize_document`), so nothing here needs a second, deferred write path.

    Always appends this run's clarifications to `silver_clarifications` — one row per (source,
    question), inserted fresh every run rather than upserted, so it accumulates a history instead
    of only the latest state — regardless of `persist`: this is the Q&A audit trail
    `agents.service.qa_pairs_for_source` reads back for `regenerate_document`/
    `ask_more_questions`, which must keep working on a still-unpublished draft.

    `mentioned_components`/`mentioned_data_contracts` are drafted once over the whole
    batch's pooled transcript (`generate_architecture_questions`), not per source — grounds
    each mention against this specific source's own content (`mentions_grounded_in_source`)
    before persisting, so a future Gold extraction pass reads a per-ADR grounded list instead
    of re-deriving it from the finished Markdown.

    Stamps `state["username"]` onto every document here, via `insert_authors_line` — AFTER
    `critic_document`/`boss_decide` have already run (this is the last node before Gold), so the
    Critic never sees this line and can't flag it as an unsupported claim. Returns the stamped
    `documents` dict regardless of `persist`, so the frontend's draft preview shows the same
    `**Authors:**` line the eventually-published version will have."""
    ingestion_date = parse_ingestion_date(state["ingestion_date"])
    documents = {
        source: insert_authors_line(content, state["username"])
        for source, content in state["documents"].items()
    }
    document_versions: dict[str, int] = dict(state["document_versions"])
    async with async_session_factory() as session:
        for source, content in documents.items():
            if state["persist"]:
                source_text = source_content(state["bronze_documents"], source)
                document_versions[source] = await _persist_document_version(
                    session,
                    ingestion_date,
                    source,
                    content,
                    mentions_grounded_in_source(source_text, state["mentioned_components"]),
                    mentions_grounded_in_source(source_text, state["mentioned_data_contracts"]),
                    state["tenant"],
                )
                _write_adr_audit_file(state["ingestion_date"], source, content)

            for item in state["clarifications"]:
                session.add(
                    SilverClarification(
                        tenant=state["tenant"],
                        ingestion_date=ingestion_date,
                        source_component=source,
                        question=item["question"],
                        answer=item["answer"],
                    )
                )
        await session.commit()

    return {"documents": documents, "document_versions": document_versions}


def route_after_write_document(state: SilverState) -> Literal["chunk_and_embed", "skip_to_end"]:
    """Gold (`chunk_and_embed` onward) only ever runs for a run that's actually persisting its
    own document — see `SilverState.persist`'s own docstring. A frontend draft ends right here,
    with nothing in Silver/Gold beyond the `silver_clarifications` audit trail
    `write_document` always writes."""
    return "chunk_and_embed" if state["persist"] else "skip_to_end"


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
                    SilverChunk.tenant == state["tenant"],
                    SilverChunk.source_component == source,
                    SilverChunk.version == version,
                )
            )
            [embedding] = await asyncio.to_thread(embed, [content])
            session.add(
                SilverChunk(
                    tenant=state["tenant"],
                    ingestion_date=ingestion_date,
                    source_component=source,
                    version=version,
                    content=content,
                    embedding=embedding,
                )
            )
        await session.commit()
    return {}


# --- Gold (.tmp/gold_process_v5.md §1-2) ----------------------------------------


async def extract_gold_facts(state: SilverState) -> dict:
    """One structured-extraction LLM call per source that has a written document this run
    (`state["documents"]` — every source, not just `active_sources`, which only holds the
    redraft target once a Boss escalation narrows it) whose `boss_verdicts` is `"ok"` —
    skipped entirely for a source still mid-redraft (never extract from an unapproved draft,
    though by the time this node runs every source in `state["documents"]` should already be
    approved; the check is defensive) and for a source whose exact `(source_component,
    version)` Gold has already processed (`gold_service.already_extracted`), so a re-run of
    this graph for an unchanged transcript spends no extra LLM call here either.

    Reads only `state["documents"][source]` — the final, clarified, boss-approved ADR — and
    nothing else. Gold used to also pass this source's grounded `mentioned_components`/
    `mentioned_data_contracts` (drafted by `generate_architecture_questions`, BEFORE any
    clarification happened) as a fixed list extraction couldn't go beyond; that meant a
    component introduced only through a clarification answer, never named in that
    pre-clarification list, could never reach Gold even though the final ADR plainly described
    it. Now Gold discovers every component/contract straight from the ADR text itself — the
    ADR is already the validated, enriched source of truth by the time this node runs, so it
    needs no earlier list to ground against (see `build_gold_extraction_prompt`'s own
    docstring)."""
    gold_extractions = dict(state["gold_extractions"])
    async with async_session_factory() as session:
        # Every source that had a document written this run, not just `active_sources` — the
        # latter shrinks to just the redrafted source once a Boss escalation happens (see
        # `boss_decide`), while `write_document`/`chunk_and_embed` already persisted/embedded
        # every OTHER already-approved source in the same batch. Iterating `active_sources`
        # here silently dropped those sources' Gold extraction for this run (bug found in
        # review — they'd stay un-extracted until an unrelated future content change bumped
        # their version again, or a manual `scripts/backfill_gold.py` run).
        for source in state["documents"]:
            if state["boss_verdicts"].get(source) != "ok":
                continue
            version = state["document_versions"][source]
            if await gold_service.already_extracted(session, source, version, tenant=state["tenant"]):
                continue

            result = await gold_service.extract_gold_facts_for_source(state["documents"][source])
            gold_extractions[source] = result.model_dump()

    return {"gold_extractions": gold_extractions}


async def resolve_gold_identity(state: SilverState) -> dict:
    """No LLM. Exact match on `gold_aliases.alias` -> `pg_trgm` fuzzy match -> mint a new
    `entity_id` (`gold_service.resolve_entity_id`), for every name this pass's extractions
    touch: each component/contract's own name, and every raw name in its
    `dependency_names`/`contract_names` — those cross-references need resolving too, so
    `persist_gold_evolution` can build `payload.dependency_ids`/`contract_ids` from
    `entity_id`s, never raw names (`ComponentPayload`, per v6 §5).

    Builds one `name -> entity_id` map per source rather than resolving inline in
    `persist_gold_evolution` — keeps identity resolution and versioning as two separable
    concerns, matching `gold_process.md` §3's own split between §3 (identity) and §5
    (versioning)."""
    entity_id_maps = dict(state["gold_entity_ids"])

    async with async_session_factory() as session:
        for source, extraction in state["gold_extractions"].items():
            version = state["document_versions"][source]
            name_to_id: dict[str, str] = dict(entity_id_maps.get(source, {}))

            for component in extraction["components"]:
                if component["status"] == "unknown":
                    continue
                await gold_service.resolve_and_alias(
                    session, "component", component["name"], name_to_id, source, version, tenant=state["tenant"]
                )
                for dep_name in component.get("dependency_names", []):
                    await gold_service.resolve_and_alias(
                        session, "component", dep_name, name_to_id, source, version, tenant=state["tenant"]
                    )
                for contract_name in component.get("contract_names", []):
                    await gold_service.resolve_and_alias(
                        session, "data_contract", contract_name, name_to_id, source, version, tenant=state["tenant"]
                    )

            for contract in extraction["contracts"]:
                if contract["action"] == "unknown":
                    continue
                await gold_service.resolve_and_alias(
                    session, "data_contract", contract["name"], name_to_id, source, version, tenant=state["tenant"]
                )

            entity_id_maps[source] = name_to_id
        await session.commit()

    return {"gold_entity_ids": entity_id_maps}


async def _persist_components(
    session, extraction: dict, name_to_id: dict, source: str, version: int, ingestion_date, tenant: str
) -> None:
    for component in extraction["components"]:
        if component["status"] == "unknown":
            continue
        # `if n in name_to_id` is unreachable today, not a real filter: both this loop and
        # `resolve_gold_identity` iterate the identical `state["gold_extractions"]` with the
        # identical `status == "unknown"` skip, so every dependency_name/contract_name reachable
        # here was already resolved into name_to_id there. Kept as a guard against the two loops'
        # coverage ever drifting apart in a future change — if it ever fires, that's the signal.
        # Sorted, not just in whatever order the LLM listed them: `_entity_hash` canonicalizes
        # dict key order but not list element order, so an unordered list here would hash
        # differently between two logically-identical extractions — a spurious version bump.
        payload = ComponentPayload(
            dependency_ids=sorted(
                {name_to_id[n] for n in component.get("dependency_names", []) if n in name_to_id}
            ),
            contract_ids=sorted(
                {name_to_id[n] for n in component.get("contract_names", []) if n in name_to_id}
            ),
        ).model_dump()
        await gold_service.persist_entity_version(
            session,
            entity_type="component",
            entity_id=name_to_id[component["name"]],
            canonical_name=component["name"],
            operation=component["status"],
            narrative=component["narrative"],
            payload=payload,
            source_component=source,
            source_adr_version=version,
            ingestion_date=ingestion_date,
            tenant=tenant,
        )


async def _persist_contracts(
    session, extraction: dict, name_to_id: dict, source: str, version: int, ingestion_date, tenant: str
) -> None:
    for contract in extraction["contracts"]:
        if contract["action"] == "unknown":
            continue
        payload = DataContractPayload(
            producer=contract["producer"],
            consumer=contract["consumer"],
            odcs_spec=gold_service.parse_odcs_spec(contract.get("odcs_spec", "")),
        ).model_dump()
        await gold_service.persist_entity_version(
            session,
            entity_type="data_contract",
            entity_id=name_to_id[contract["name"]],
            canonical_name=contract["name"],
            operation=contract["action"],
            narrative=contract["narrative"],
            payload=payload,
            source_component=source,
            source_adr_version=version,
            ingestion_date=ingestion_date,
            tenant=tenant,
        )


async def _persist_architecture(
    session, extraction: dict, source: str, version: int, ingestion_date, tenant: str
) -> None:
    """Scoped **per source_component**, not one global "architecture as a whole" entity across
    every source in the batch — reconciling multiple sources' architecture views into one entity
    is cross-source coordination, which `.tmp/gold_process_v5.md` §6 explicitly keeps out of a
    single graph run for this pass. Each source gets its own `architecture:<source_component>`
    entity_id, versioned independently; merging them into one global view is future work."""
    # Sorted for the same reason as ComponentPayload's dependency_ids/contract_ids — order must
    # not affect `_entity_hash`, and what's persisted should be order-stable too.
    payload = ArchitecturePayload(
        mermaid_diagram=extraction.get("mermaid_diagram", ""),
        components=sorted({c["name"] for c in extraction["components"]}),
        dependencies=sorted({dep for c in extraction["components"] for dep in c.get("dependency_names", [])}),
    ).model_dump()
    await gold_service.persist_entity_version(
        session,
        entity_type="architecture",
        entity_id=f"architecture:{source}",
        canonical_name=source,
        operation=extraction["architecture_change"],
        narrative=extraction["architecture_narrative"],
        payload=payload,
        source_component=source,
        source_adr_version=version,
        ingestion_date=ingestion_date,
        tenant=tenant,
    )


async def persist_gold_evolution(state: SilverState) -> dict:
    """No LLM. Hash-compare-then-bump per entity (`gold_service.persist_entity_version`) —
    components and data contracts whose `status`/`action` is `"unknown"` are skipped entirely,
    never written as a `gold_evolution` row (v6 §3: an unresolved classification is a signal the
    extraction is under-grounded, not a valid value to version). One helper per entity kind —
    `_persist_components`/`_persist_contracts`/`_persist_architecture` — since each builds a
    differently-shaped payload; see `_persist_architecture`'s own docstring for why architecture
    is scoped per source_component rather than globally."""
    ingestion_date = parse_ingestion_date(state["ingestion_date"])
    async with async_session_factory() as session:
        for source, extraction in state["gold_extractions"].items():
            version = state["document_versions"][source]
            name_to_id = state["gold_entity_ids"].get(source, {})
            tenant = state["tenant"]
            await _persist_components(session, extraction, name_to_id, source, version, ingestion_date, tenant)
            await _persist_contracts(session, extraction, name_to_id, source, version, ingestion_date, tenant)
            await _persist_architecture(session, extraction, source, version, ingestion_date, tenant)
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
    graph.add_node("extract_gold_facts", extract_gold_facts)
    graph.add_node("resolve_gold_identity", resolve_gold_identity)
    graph.add_node("persist_gold_evolution", persist_gold_evolution)

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
    graph.add_conditional_edges(
        "write_document",
        route_after_write_document,
        {"chunk_and_embed": "chunk_and_embed", "skip_to_end": END},
    )
    graph.add_edge("chunk_and_embed", "extract_gold_facts")
    graph.add_edge("extract_gold_facts", "resolve_gold_identity")
    graph.add_edge("resolve_gold_identity", "persist_gold_evolution")
    graph.add_edge("persist_gold_evolution", END)

    return graph.compile(checkpointer=checkpointer)
