"""The Silver clarification loop, from start to end. This turns `doc/silver_process.md` §3
into code.

This is one large module on purpose, instead of one file per node. The ten nodes below only
make sense when wired together. See `.tmp/tasks.md` Task 6 for why splitting this into
separate commits would mean landing broken states in between.
"""

import os

# This must run before `langgraph` (imported below) loads anywhere in the process. LangGraph
# ships its own OpenTelemetry tracing through the bundled LangSmith SDK, and it reads these
# settings at import time. Once `logfire.configure()` below installs the global OTel tracer
# provider, LangSmith's tracer detects it. After that, every graph-run and node span flows
# straight into Logfire. We need no separate LangSmith account, exporter, or API key for this.
os.environ.setdefault("LANGSMITH_OTEL_ENABLED", "true")
os.environ.setdefault("LANGSMITH_OTEL_ONLY", "true")
os.environ.setdefault("LANGSMITH_TRACING", "true")

import asyncio
import re
from pathlib import Path
from typing import Literal

import logfire
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt
from sqlalchemy import delete, select

from agents.shared import (
    SHALLOW_RETRY_ATTEMPTS,
    SHALLOW_RETRY_TEMPERATURE,
    distinct_sources,
    extract_authors_line,
    insert_authors_line,
    latest_document_content,
    load_bronze_rows,
    mentions_grounded_in_source,
    source_content,
    transcription_base_name,
)
from agents.stages import gold
from agents.stages.adr_critic.prompts import build_critic_prompt
from agents.stages.adr_critic.schemas import CritiqueResult
from agents.stages.adr_generation.prompts import build_adr_generation_prompt
from agents.stages.adr_generation.service import adr_has_placeholder_leak, strip_diagram_colors
from agents.stages.architecture_questions.service import (
    generate_architecture_questions_for_batch,
    previous_architecture_context,
)
from agents.stages.classification.prompts import build_classification_prompt
from agents.stages.classification.schemas import ClassificationResult, QuestionClassification
from agents.stages.data_contract_questions.service import generate_data_contract_questions_for_batch
from agents.stages.gold.schemas import ArchitecturePayload, ComponentPayload, DataContractPayload
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

# The frontend's per-question "Infer an answer" and "Suggest info" buttons (see
# `frontend/app.py`) send one of these two exact strings as the answer, instead of free text.
# We never add these to `_DECLINE_PHRASES` above. They must survive into
# `SilverClarification.answer` as real answered text, for
# `prompts/adr_generation/generator.jinja`'s own CLARIFICATION ANSWER MARKERS section to read.
# "Irrelevant" is different: it really does mean "no answer," so it belongs in the decline set
# instead. It has no special meaning downstream of its own.
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
    """When `scopes` is given (a map from question text to scope), the questions came from
    the classify stage and each one carries a real scope. In that case, this function splits
    the questions into two buckets: an architecture bucket (everything except
    `data_contract`, ranked by `_QUESTIONS_PRIORITIES` and capped at `architecture_cap`) and a
    data-contract bucket (capped at `data_contract_cap`, cut off in drafted order). Boss-origin
    escalations (see `boss_decide`) have no scope of their own. They are made up on the spot
    from a Critic claim, and there are already naturally few of them. So without `scopes`, we
    just cut them down to `architecture_cap` in their existing order.

    Both caps come from the run's own state
    (`state["max_architecture_pending_questions"]` and
    `state["max_data_contract_pending_questions"]`). We do not read them from `settings`
    directly here. See `agents.state.initial_state` for where a caller — the frontend's
    upload form, a script, or a test — can override them. Otherwise they fall back to
    `settings`'s own defaults.

    This removes duplicates by exact text first, keeping the first-seen order. Two identical
    question strings can occur — for example, `boss_decide` quoting the same Critic claim
    text twice, or the classify stage drafting the same wording twice. These must never reach
    `ask_human` as two separate pending questions. This node's own resume payload, and the
    frontend's one-widget-per-question display, both key off the question text. So a real
    duplicate would either silently collapse into one answer, or crash the frontend outright,
    since Streamlit requires unique widget keys."""
    questions = list(dict.fromkeys(questions))
    if scopes is None:
        return questions[:architecture_cap]

    architecture_qs = [q for q in questions if scopes.get(q) != "data_contract"]
    data_contract_qs = [q for q in questions if scopes.get(q) == "data_contract"]
    architecture_qs.sort(key=lambda q: _QUESTIONS_PRIORITIES.get(scopes.get(q, ""), len(_QUESTIONS_PRIORITIES)))

    return architecture_qs[:architecture_cap] + data_contract_qs[:data_contract_cap]


def checkpointer_dsn() -> str:
    """`AsyncPostgresSaver` needs psycopg's DSN format, not SQLAlchemy's `+asyncpg` one. It is
    still the same `technical_meeting_rag` database either way. No new infrastructure is
    needed."""
    return settings.database_url.replace("postgresql+asyncpg://", "postgresql://")


def _is_inside_code_fence(content: str, index: int) -> bool:
    """Returns `True` if `index` falls inside a ` ``` `-fenced block. An odd number of fence
    markers before `index` means we are currently between an opening fence and a closing one."""
    return content.count("```", 0, index) % 2 == 1


def _downgrade_claim(content: str, claim: str) -> str:
    """Adds `_DOWNGRADE_MARKER` right after the first occurrence of `claim` that is NOT
    inside a ` ```mermaid ` fence. The ADR's own "previous" and "target architecture"
    diagrams live in fenced blocks (see `prompts/adr_generation/generator.jinja`'s OUTPUT
    STRUCTURE). The Critic's claim text is a word-for-word quote from anywhere in the
    document, including diagram node labels — `prompts/adr_critic/critic.jinja`'s own quoting
    rule does not exempt them. Splicing `**[unknown — flagged by review]**` into a Mermaid
    label breaks its syntax. Mermaid does not know what to do with a stray `**`, so the
    rendered diagram fails outright instead of just looking annotated. If `claim` only ever
    occurs inside a fence, this function returns `content` unchanged instead of corrupting
    the diagram. In that case there is no prose sentence to annotate, but the diagram must
    still render."""
    if not claim:
        return content
    marked = claim + _DOWNGRADE_MARKER
    if marked in content:
        return content  # Already downgraded. Running this again does nothing new.

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
    """Generates architecture questions and mentions.

    `architecture_diagram` (this stage's own KNOWN_ARCHITECTURE input) is built from each
    distinct source's own previous ADR, if one exists. This comes from Silver's own version
    history, never from Gold (see `agents.stages.architecture_questions.service.previous_architecture_context`). A source
    with no prior ADR contributes nothing here. That is exactly "no prior architecture known"
    from the prompt's point of view."""
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
    # A name the LLM put in both lists is really a data contract wearing a component's
    # clothes, not two separate entities. "Topic" or "Queue" is a valid component type
    # (architecture_questions.jinja PHASE 3), and a data contract is often named after the
    # topic or queue it travels over. So the two lists can genuinely collide on the same
    # name, even with the prompt's own "don't double-list" instruction — that instruction is
    # still just prompt-following, not something we enforce. We prefer the data contract
    # classification here mechanically, not by asking the LLM to try harder. This way, every
    # downstream consumer (grounding checks, Gold extraction, the ADR itself) sees one
    # consistent classification, instead of each one deciding on its own.
    contract_names = {c.name.lower() for c in result.mentioned_data_contracts}
    components = [c for c in result.mentioned_components if c.name.lower() not in contract_names]
    return {
        "generated_questions": [q.model_dump() for q in result.questions],
        "mentioned_components": [c.model_dump() for c in components],
        "mentioned_data_contracts": [c.model_dump() for c in result.mentioned_data_contracts],
    }


def _slugify(name: str) -> str:
    """Lowercases the name and collapses runs of non-alphanumeric characters into one
    underscore. This matches the `contract.<slug>.schema` id convention that
    `prompts/data_contract_questions/questions.jinja`'s PHASE 14 already defines. This way, a
    fallback question's `id` reads like one the LLM drafted, not like a made-up marker."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "contract"


def _ensure_schema_questions(questions: list[dict], mentioned_data_contracts: list[dict]) -> list[dict]:
    """This is a mechanical backstop for `prompts/data_contract_questions/questions.jinja`'s
    own FINAL SELF-CHECK, which says schema fields must be known or questioned for every
    contract in IDENTIFIED_DATA_CONTRACTS. That self-check only works if the LLM follows the
    prompt. Nothing downstream catches a contract where the LLM drafted OTHER questions
    (version, ownership, SLA, and so on) but never actually asked what its schema contains. A
    data contract with no fields on record can never get a real ODCS spec at Gold-extraction
    time — `gold_extraction.jinja`'s `odcs_spec` would just stay `"{}"`. So every identified
    contract MUST have a schema question drafted for it. This is a real requirement, not a
    nice-to-have, which is why we enforce it here instead of leaving it to the prompt alone.

    This only adds a schema question if none already exists for that contract. A match is
    found by `target` plus the word "schema" appearing somewhere in the drafted `id`, the
    PHASE 14 convention. `classify_questions` still decides whether the transcript already
    answers this question, the same as any other question. This function only stops the
    question from being silently skipped in the first place."""
    targets_with_schema_question = {
        q["target"] for q in questions if q.get("scope") == "data_contract" and "schema" in q.get("id", "")
    }
    result = list(questions)
    for contract in mentioned_data_contracts:
        name = contract["name"]
        if name in targets_with_schema_question:
            continue
        result.append(
            {
                "id": f"contract.{_slugify(name)}.schema",
                "scope": "data_contract",
                "target": name,
                "requirement": "schema",
                "question": (
                    f"What fields (with type, required/nullable status, and meaning) make up "
                    f"the {name} data contract's schema?"
                ),
            }
        )
    return result


def _drop_new_contract_version_questions(questions: list[dict], mentioned_data_contracts: list[dict]) -> list[dict]:
    """This is a mechanical backstop for `prompts/data_contract_questions/questions.jinja`'s
    own PHASE 3 rule: a brand-new contract's version is always `1.0.0`, a fixed convention, not
    something anyone states in a meeting and not something worth asking about.
    `prompts/adr_generation/generator.jinja` already treats it the same way when it writes the
    ADR. That rule only holds if the LLM follows the prompt — nothing stops it from drafting a
    version question for a `new` contract anyway, the same reason `_ensure_schema_questions`
    above exists as a backstop instead of trusting the prompt alone. This drops any drafted
    question whose id names a `.version`-shaped field (the PHASE 14 convention) for a contract
    whose `action` (from the architecture stage) is `new`."""
    new_contract_names = {c["name"] for c in mentioned_data_contracts if c.get("action") == "new"}
    return [
        q for q in questions if not (q.get("target") in new_contract_names and "version" in q.get("id", ""))
    ]


async def generate_data_contract_questions(state: SilverState) -> dict:
    """Generates data contract questions and mentions."""
    result = await generate_data_contract_questions_for_batch(
        state["ingestion_date"], state["bronze_documents"], state["mentioned_data_contracts"]
    )
    contract_questions = _ensure_schema_questions(
        [q.model_dump() for q in result.questions], state["mentioned_data_contracts"]
    )
    contract_questions = _drop_new_contract_version_questions(contract_questions, state["mentioned_data_contracts"])
    return {
        "generated_questions": state["generated_questions"] + contract_questions,
    }


def _canonical_classification(
    classification_id: str,
    by_classification_id: dict[str, QuestionClassification],
    seen: frozenset[str],
) -> QuestionClassification:
    """Follows one question's `duplicate_of` chain back to the question that actually carries
    the real status and answer. `duplicate_of` marks a question the classifier judged as
    asking for the same information as another one in this same batch, in different words —
    see `prompts/classification/classifier.jinja`'s DUPLICATE QUESTIONS section. Stops and
    returns this question's own classification unchanged if `duplicate_of` is empty, points at
    itself, points at an id outside this batch, or would revisit an id already seen (a
    malformed loop the classifier should not produce, but must not hang on if it does)."""
    classification = by_classification_id[classification_id]
    target_id = classification.duplicate_of
    if target_id is None or target_id == classification_id or target_id in seen or target_id not in by_classification_id:
        return classification
    return _canonical_classification(target_id, by_classification_id, seen | {classification_id})


async def classify_questions(state: SilverState) -> dict:
    """This is the single decision that gates every question this graph run will ever show a
    human. `answered` silently drops a question. `needs_clarification` keeps it. We use
    `temperature=0` here because this call's own randomness, not the drafted questions'
    quality, was the real root cause of a reported bug. A healthy, well-drafted question set
    was collapsing to 1-2 surfaced questions from one run to the next. This happened because
    a non-zero-temperature classify call sometimes over-inferred "answered" from prose that
    sounded confident but did not actually commit to anything. `reasoning_effort="none"` goes
    alongside it, for the same reasoning-locked-model workaround that
    `generate_architecture_questions_for_batch` already explains in its own docstring.

    This same call also finds semantic duplicates among the drafted questions — two questions
    worded differently that ask for the same information (see `prompts/classification/
    classifier.jinja`'s DUPLICATE QUESTIONS section) — and folds each duplicate into the
    question it duplicates, via `_canonical_classification` below."""
    questions = state["generated_questions"]
    messages = build_classification_prompt(questions, state["transcript_text"])
    response = await router.complete(
        messages, response_format=ClassificationResult, temperature=0, reasoning_effort="none"
    )
    result = ClassificationResult.model_validate(load_json_response(response.choices[0].message.content))

    # We match this back to the drafted question by `id`, not by the resent text. An id the
    # classifier made up, one that was not among the questions it was given, gets dropped
    # instead of crashing the node on a malformed response.
    by_id = {q["id"]: q for q in questions}
    by_classification_id = {c.id: c for c in result.classifications}
    clarifications: list[ClarificationItem] = []
    for c in result.classifications:
        question = by_id.get(c.id)
        if question is None:
            continue
        # A duplicate question is folded into the one it duplicates: same question text, same
        # status, same answer. `_top_questions` below already collapses exact-text duplicates
        # into one pending question, and `ask_human` already fills every clarification that
        # shares that exact text when the human answers it once — so a duplicate needs no
        # separate handling past this point. `canonical` is `c` itself when this question is
        # not a duplicate of anything.
        canonical = _canonical_classification(c.id, by_classification_id, frozenset())
        canonical_question = by_id.get(canonical.id, question)
        clarifications.append(
            {
                "id": question["id"],
                "scope": question["scope"],
                "target": question["target"],
                "requirement": question["requirement"],
                "question": canonical_question["question"],
                "answer": canonical.answer,
                "status": canonical.status,
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
    """This sends one `interrupt()` carrying every pending question together. We reuse it for
    both the classify-stage gap-filling (`interrupt_origin == "classify"`) and a Boss
    escalation over a contradiction (`interrupt_origin == "boss"`). It is the same mechanism
    and the same Postgres-backed checkpointer. There is no separate pause path for the second
    case."""
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
            # This is a Boss-origin escalation question (see boss_decide). It is made up on
            # the spot from a Critic claim, so it never went through generate_questions and
            # has no real id, scope, or target of its own. We fill in placeholders that still
            # say what it is about, so this item's shape matches every other one.
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
    """This is the Actor. It makes one LLM call per distinct source, writing the final ADR
    directly, following `prompts/adr_generation/generator.jinja`'s own structure
    (`doc/silver_process.md` §3 node 6). It takes a flat question/answer list, not the
    graph's own richer `ClarificationItem` shape — see `agents.stages.adr_generation.prompts.QaPair`. So `id`,
    `scope`, `target`, and `requirement` are dropped here. The ADR prompt only ever reads the
    question text and its answer.

    §2 "Previous Architecture" is grounded in Gold's own current state
    (`gold.current_architecture_diagram`), not in Silver's per-source document history. Gold
    is the reconciled, cross-source, entity-resolved record of what has actually been
    published so far. So an ADR for source A correctly shows components that a DIFFERENT
    source's earlier ADR already established, not just source A's own history. This is
    computed once per run, not once per source in `sources`: Gold's architecture is one
    tenant-wide state, not a per-source one, so every source in this batch sees the exact
    same "previous" snapshot. We call `strip_diagram_colors` because Gold's own diagram
    carries its `goldNode` display styling. §2 must always be a plain, colorless snapshot,
    the same rule that `previous_target_architecture_diagram` and
    `own_previous_architecture_diagram` already applied when this read from Silver instead.
    This is empty only when Gold has no live component for this tenant yet, meaning a true
    first-ever run."""
    sources = state["redraft_only"] or distinct_sources(state["bronze_documents"])
    qa_pairs = [{"question": c["question"], "answer": c["answer"]} for c in state["clarifications"]]

    documents = dict(state["documents"])
    async with async_session_factory() as session:
        previous_diagram = strip_diagram_colors(
            await gold.current_architecture_diagram(session, tenant=state["tenant"])
        )
        for source in sources:
            messages = build_adr_generation_prompt(
                source_content(state["bronze_documents"], source), qa_pairs, previous_diagram
            )
            # We use temperature=0 for the same reason `classify_questions` needs it: we saw
            # a real run-to-run inconsistency here, where the same case's structural
            # checklist passed and then failed across two identical `agents.stages.adr_generation.testing` runs,
            # with nothing about the input having changed. `reasoning_effort="none"` is the
            # matching workaround a reasoning-locked model needs alongside temperature=0 —
            # see `generate_architecture_questions_for_batch`'s own docstring.
            response = await router.complete(messages, temperature=0, reasoning_effort="none")
            content = response.choices[0].message.content
            # This is a mechanical retry on a template leak (see `adr_has_placeholder_leak`).
            # It follows the same shallow-retry approach that
            # `generate_architecture_questions_for_batch` already uses for ITS stage's own
            # observed failure, applied here for ADR generation's own failure: temperature=0
            # makes a clean run reliably reproducible, but it also reliably reproduces a leak
            # if one happens. Each retry samples at a nonzero temperature to break out of
            # that fixed behavior. It is not a general "retry on any failure" policy.
            for _ in range(SHALLOW_RETRY_ATTEMPTS):
                if not adr_has_placeholder_leak(content):
                    break
                retry_response = await router.complete(
                    messages, temperature=SHALLOW_RETRY_TEMPERATURE, reasoning_effort="none"
                )
                content = retry_response.choices[0].message.content
            documents[source] = content

    return {
        "documents": documents,
        "active_sources": sources,
        "redraft_only": None,
    }


async def critic_document(state: SilverState) -> dict:
    """This node is mandatory. It always runs, with no confidence threshold that skips it. It
    also produces this source's `completeness_score` and `unresolved_points` (see
    `prompts/adr_critic/critic.jinja`'s COMPLETENESS SCORING section) from the same read. We
    need no second LLM call just to grade the document separately from reviewing its claims."""
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
    """This is deterministic. It makes no LLM call. It applies a policy over the Critic's
    output, rather than giving a second opinion. It escalates a real contradiction once, and
    downgrades everything else."""
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
    """Writes one `SilverDocument` row, deciding the version from the content alone, never
    from what the LLM says about itself. This compares the new ADR's hash against the latest
    existing row for this `source_component`. An identical hash overwrites that same row in
    place — this is an idempotent re-run, with no new version. A different hash inserts a new
    row at `latest_version + 1`, leaving the older version's row untouched. Both rows stay
    queryable. This returns the version actually written, so `chunk_and_embed` does not have
    to work it out again.

    `mentioned_component_names` and `mentioned_data_contract_names` get refreshed on every
    write, including on the same-hash overwrite branch. A re-run can genuinely draft a
    different, or differently grounded, mention list, even when the synthesized ADR text
    itself hashes the same. `_row_fields` is the single place where all three branches
    (insert-first, overwrite, insert-next-version) read shared column values from. This way,
    adding a column later is a one-line change here, instead of a hand-edit repeated across
    three constructor calls.

    `authored_by` is read straight out of `content`'s own `**Authors:**` line
    (`agents.shared.extract_authors_line`), never as a separate parameter. This gives this
    function exactly one source of truth for who authored a version, matching whatever the
    document itself says. This is true even on the overwrite branch: an identical-hash re-run
    still refreshes it, for the same reason as the mention lists above."""
    new_hash = gold.content_hash(content)
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
    """Writes `output/ingestion_date=<date>/adr/<transcription>.md`. This is a human-
    readable audit copy of the ADR that `write_document` just saved, following the same
    convention that `agents.shared._write_json_audit_file` already uses for the two
    question-generation stages' own audit files (`doc/silver_process.md` §1). This always
    shows this run's latest content, and it has no version history on disk. Only
    `silver_documents` keeps version history. This file exists for a human to read, not for
    anything downstream to read."""
    adr_dir = Path(settings.output_dir) / f"ingestion_date={ingestion_date_str}" / "adr"
    adr_dir.mkdir(parents=True, exist_ok=True)
    (adr_dir / f"{transcription_base_name(source_component)}.md").write_text(content, encoding="utf-8")


async def write_document(state: SilverState) -> dict:
    """Saves each synthesized ADR at its own `(source_component, version)` pair — see
    `_persist_document_version` for how it decides between overwriting and starting a new
    version. It also writes each ADR to disk (`_write_adr_audit_file`) for a person to check
    by hand, alongside the Postgres write. This only happens when `state["persist"]` is true.
    The frontend's initial upload runs with it false, so this ADR, and everything downstream
    of it — see `route_after_write_document` — stays a draft until a human clicks "Publish."
    Publishing later calls `_persist_document_version` itself (`finalize_document`), so
    nothing here needs a second, deferred write path.

    This always adds this run's clarifications to `silver_clarifications`, one row per
    (source, question), inserted fresh every run rather than updated in place. This way it
    builds up a history instead of only keeping the latest state. This happens regardless of
    `persist`. This is the question-and-answer audit trail that
    `agents.shared.qa_pairs_for_source` reads back for `regenerate_document` and
    `ask_more_questions`, which must keep working on a still-unpublished draft.

    `mentioned_components` and `mentioned_data_contracts` are drafted once over the whole
    batch's pooled transcript (`generate_architecture_questions`), not once per source. This
    function checks each mention against this specific source's own content
    (`mentions_grounded_in_source`) before saving it. That way, a future Gold extraction pass
    reads a per-ADR grounded list, instead of working it out again from the finished
    Markdown.

    This stamps `state["username"]` onto every document here, using
    `insert_authors_line`, AFTER `critic_document` and `boss_decide` have already run — this
    is the last node before Gold. So the Critic never sees this line, and cannot flag it as
    an unsupported claim. This returns the stamped `documents` dict regardless of `persist`,
    so the frontend's draft preview shows the same `**Authors:**` line that the eventually
    published version will have."""
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
    """Gold (`chunk_and_embed` onward) only ever runs for a run that is actually saving its
    own document — see `SilverState.persist`'s own docstring. A frontend draft ends right
    here, with nothing written to Silver or Gold beyond the `silver_clarifications` audit
    trail that `write_document` always writes."""
    return "chunk_and_embed" if state["persist"] else "skip_to_end"


async def chunk_and_embed(state: SilverState) -> dict:
    """This makes one embedding per ADR, with no token-splitting: an ADR is retrieved as a
    whole, never as a fragment (`doc/silver_process.md` §3 node 10). This updates exactly the
    `(source_component, version)` pair that `write_document` just wrote, leaving every other
    version's chunk row (older versions, other sources, other dates) untouched. This is the
    opposite of the old approach, which deleted everything by `ingestion_date` in one go —
    that old approach would have wiped out the version history that this node now exists to
    keep. `END`."""
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
    """This makes one structured-extraction LLM call per source that meets two conditions:
    it has a document written this run (`state["documents"]` — every source, not just
    `active_sources`, which only holds the redraft target once a Boss escalation narrows it
    down), and its `boss_verdicts` value is `"ok"`. This is skipped entirely for a source
    still mid-redraft. We never extract from an unapproved draft, though by the time this
    node runs, every source in `state["documents"]` should already be approved — this check
    is just a safety net. It is also skipped for a source whose exact
    `(source_component, version)` Gold has already processed (`gold.already_extracted`), so a
    re-run of this graph for an unchanged transcript spends no extra LLM call here either.

    This reads only `state["documents"][source]`, the final, clarified, Boss-approved ADR,
    and nothing else. Gold used to also take this source's grounded `mentioned_components`
    and `mentioned_data_contracts` (drafted by `generate_architecture_questions`, BEFORE any
    clarification happened) as a fixed list that extraction could not go beyond. That meant a
    component introduced only through a clarification answer, one never named in that
    pre-clarification list, could never reach Gold, even though the final ADR plainly
    described it. Now Gold finds every component and contract straight from the ADR text
    itself. By the time this node runs, the ADR is already the validated, enriched source of
    truth, so it needs no earlier list to check against (see
    `build_gold_extraction_prompt`'s own docstring)."""
    gold_extractions = dict(state["gold_extractions"])
    async with async_session_factory() as session:
        # This loops over every source that had a document written this run, not just
        # `active_sources`. `active_sources` shrinks to just the redrafted source once a
        # Boss escalation happens (see `boss_decide`), while `write_document` and
        # `chunk_and_embed` already saved or embedded every OTHER already-approved source in
        # the same batch. Looping over `active_sources` here silently dropped those other
        # sources' Gold extraction for this run — this bug was found in review. Those
        # sources would then stay un-extracted until an unrelated future content change
        # bumped their version again, or until someone ran `scripts/backfill_gold.py` by
        # hand.
        for source in state["documents"]:
            if state["boss_verdicts"].get(source) != "ok":
                continue
            version = state["document_versions"][source]
            if await gold.already_extracted(session, source, version, tenant=state["tenant"]):
                continue

            result = await gold.extract_gold_facts_for_source(state["documents"][source])
            gold_extractions[source] = result.model_dump()

    return {"gold_extractions": gold_extractions}


async def resolve_gold_identity(state: SilverState) -> dict:
    """No LLM call happens here. This tries an exact match on `gold_aliases.alias` first,
    then a `pg_trgm` fuzzy match, and finally mints a new `entity_id` (`gold.resolve_entity_id`)
    if neither match works. It does this for every name this pass's extractions touch: each
    component's or contract's own name, plus every raw name in its `dependency_names` or
    `contract_names`, plus a contract's own `producer`/`consumer` names. Those cross-references
    need resolving too, so that `persist_gold_evolution` can build `payload.dependency_ids`,
    `payload.contract_ids`, and a contract's own `payload.producer_id`/`consumer_id` from
    `entity_id`s, never from raw names (`ComponentPayload`/`DataContractPayload`, per v6 §5). A
    contract's producer and consumer are themselves components — `resolve_and_alias` is called
    with `entity_type="component"` for them, the same as for `dependency_names`.

    This builds one `name -> entity_id` map per source, instead of resolving names inline
    inside `persist_gold_evolution`. This keeps identity resolution and versioning as two
    separate concerns, matching the split that `gold_process.md` §3 already makes between §3
    (identity) and §5 (versioning)."""
    entity_id_maps = dict(state["gold_entity_ids"])

    async with async_session_factory() as session:
        for source, extraction in state["gold_extractions"].items():
            version = state["document_versions"][source]
            name_to_id: dict[str, str] = dict(entity_id_maps.get(source, {}))

            for component in extraction["components"]:
                if component["status"] == "unknown":
                    continue
                await gold.resolve_and_alias(
                    session, "component", component["name"], name_to_id, source, version, tenant=state["tenant"]
                )
                for dep_name in component.get("dependency_names", []):
                    await gold.resolve_and_alias(
                        session, "component", dep_name, name_to_id, source, version, tenant=state["tenant"]
                    )
                for contract_name in component.get("contract_names", []):
                    await gold.resolve_and_alias(
                        session, "data_contract", contract_name, name_to_id, source, version, tenant=state["tenant"]
                    )

            for contract in extraction["contracts"]:
                if contract["action"] == "unknown":
                    continue
                await gold.resolve_and_alias(
                    session, "data_contract", contract["name"], name_to_id, source, version, tenant=state["tenant"]
                )
                for component_name in (contract["producer"], contract["consumer"]):
                    await gold.resolve_and_alias(
                        session, "component", component_name, name_to_id, source, version, tenant=state["tenant"]
                    )

            entity_id_maps[source] = name_to_id
        await session.commit()

    return {"gold_entity_ids": entity_id_maps}


async def _persist_components(
    session, extraction: dict, name_to_id: dict, source: str, version: int, ingestion_date, tenant: str,
    authored_by: str,
) -> None:
    for component in extraction["components"]:
        if component["status"] == "unknown":
            continue
        # The `if n in name_to_id` check below never actually filters anything out today.
        # This loop and `resolve_gold_identity` both iterate the exact same
        # `state["gold_extractions"]`, with the exact same `status == "unknown"` skip. So
        # every dependency_name and contract_name reachable here was already resolved into
        # name_to_id there. We keep the check as a guard, in case these two loops' coverage
        # ever drifts apart in a future change — if it ever fires, that drift is the signal
        # to look into.
        # These are sorted, not left in whatever order the LLM listed them: `_entity_hash`
        # makes dict key order consistent, but not list element order. So an unsorted list
        # here would hash differently between two extractions that are logically identical,
        # causing a version bump for no real reason.
        payload = ComponentPayload(
            dependency_ids=sorted(
                {name_to_id[n] for n in component.get("dependency_names", []) if n in name_to_id}
            ),
            contract_ids=sorted(
                {name_to_id[n] for n in component.get("contract_names", []) if n in name_to_id}
            ),
        ).model_dump()
        await gold.persist_entity_version(
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
            authored_by=authored_by,
        )


async def _persist_contracts(
    session, extraction: dict, name_to_id: dict, source: str, version: int, ingestion_date, tenant: str,
    authored_by: str,
) -> None:
    for contract in extraction["contracts"]:
        if contract["action"] == "unknown":
            continue
        payload = DataContractPayload(
            producer=contract["producer"],
            consumer=contract["consumer"],
            producer_id=name_to_id.get(contract["producer"], ""),
            consumer_id=name_to_id.get(contract["consumer"], ""),
            odcs_spec=gold.parse_odcs_spec(contract.get("odcs_spec", "")),
        ).model_dump()
        await gold.persist_entity_version(
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
            authored_by=authored_by,
        )


async def _persist_architecture(
    session, extraction: dict, source: str, version: int, ingestion_date, tenant: str, authored_by: str
) -> None:
    """This is scoped per source_component, not to one global "architecture as a whole"
    entity across every source in the batch. Reconciling multiple sources' architecture views
    into one entity is cross-source coordination, and `.tmp/gold_process_v5.md` §6 says
    explicitly to keep that kind of coordination out of a single graph run for this pass.
    Each source gets its own `architecture:<source_component>` entity_id, versioned on its
    own. Merging these into one global view is future work."""
    # These are sorted for the same reason as ComponentPayload's dependency_ids and
    # contract_ids above: order must not affect `_entity_hash`, so what we save should stay
    # in a stable order too.
    payload = ArchitecturePayload(
        mermaid_diagram=extraction.get("mermaid_diagram", ""),
        components=sorted({c["name"] for c in extraction["components"]}),
        dependencies=sorted({dep for c in extraction["components"] for dep in c.get("dependency_names", [])}),
    ).model_dump()
    await gold.persist_entity_version(
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
        authored_by=authored_by,
    )


async def persist_gold_evolution(state: SilverState) -> dict:
    """No LLM call happens here. This compares hashes and bumps a version, per entity
    (`gold.persist_entity_version`). Components and data contracts whose `status` or
    `action` is `"unknown"` are skipped entirely. They are never written as a
    `gold_evolution` row (v6 §3: an unresolved classification is a sign that the extraction
    is under-grounded, not a valid value to version). There is one helper per entity kind —
    `_persist_components`, `_persist_contracts`, `_persist_architecture` — because each one
    builds a differently shaped payload. See `_persist_architecture`'s own docstring for why
    architecture is scoped per source_component rather than globally.

    `authored_by` is read straight out of `state["documents"][source]`'s own `**Authors:**`
    line, the same `agents.shared.extract_authors_line` call `_persist_document_version`
    already makes for `silver_documents.authored_by`. This is the exact same content
    `write_document` already persisted this run, so no extra DB read is needed to get the
    same value here."""
    ingestion_date = parse_ingestion_date(state["ingestion_date"])
    async with async_session_factory() as session:
        for source, extraction in state["gold_extractions"].items():
            version = state["document_versions"][source]
            name_to_id = state["gold_entity_ids"].get(source, {})
            tenant = state["tenant"]
            authored_by = extract_authors_line(state["documents"][source])
            await _persist_components(
                session, extraction, name_to_id, source, version, ingestion_date, tenant, authored_by
            )
            await _persist_contracts(
                session, extraction, name_to_id, source, version, ingestion_date, tenant, authored_by
            )
            await _persist_architecture(session, extraction, source, version, ingestion_date, tenant, authored_by)
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
    # ask_human always continues on to synthesize_document, no matter its origin. A
    # classify-origin resume has just resolved every pending question, so there is nothing
    # left to loop back to route_after_classify for. A boss-origin resume redrafts only the
    # flagged source (`redraft_only`, set by boss_decide before interrupting).
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
