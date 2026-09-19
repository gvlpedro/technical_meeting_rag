"""This test suite checks Gold's real-time retrieval.

It proves that agents/graph.py writes correct facts to gold_evolution.
It also proves a RAG consumer can retrieve those facts, via `top_k_gold_evolution`'s
`mode="hybrid"` (vector cosine-distance search fused with a lexical `ts_rank` search via
Reciprocal Rank Fusion — see `.tmp/advanced_techniques.md` §1) — the same retrieval strategy
`scripts/chat_gold.py` and `/v1/frontend/chat` use in production, not a plain-vector snapshot
from before hybrid search existed.

The suite sends 10 real meeting transcripts through the real graph.
Each transcript adds one month to the same architecture (2026-01-15 to 2026-10-15).
Every call is real: real LLM calls, real embeddings, real Postgres writes.
No mock replaces the LLM.

The suite walks four full state machines end to end, as two independent threads:
- Steps 01-05 (the original suite): the "Order Events" data contract moves through every
  ContractAction value (new, forward-update, break-change, deprecated, removed), while the
  "Legacy Order Monolith" component moves through every ComponentStatus value except "unknown"
  (unchanged, then removed).
- Steps 06-10 (added to double this suite's case count): the "Refund Issued" data contract
  walks the exact same full ContractAction lifecycle Order Events already walked, on a
  completely separate producer/consumer thread (Refund Service, Fraud Check Service). This
  doubles the real Gold history a retrieval-quality check has to work against, and proves the
  same lifecycle-tracking correctness holds for a second, independent entity — not just once,
  by coincidence.

Read golden_set/<step>/transcript.txt for the full story of each step.

Each step runs two kinds of check:
- Deterministic checks. These checks gate the test. One check compares
  gold_evolution rows against golden_set/<step>/expected_gold_facts.json.
  Another check compares retrieval and answer results against
  golden_set/<step>/qa.json.
- A suite-level checklist. This check runs once, after all 10 steps finish.
  See checklist.txt for the full list of items. Five items are deterministic
  and gate the test. One item is a judgment call from a second LLM. That item
  is informational only. It does not gate the test.

pyproject.toml excludes this suite from `make test`. Each step sends several
real LLM calls on purpose. Run this suite on its own:

    make test-gold-arch-evolution
"""

import json
from datetime import date
from pathlib import Path
from typing import get_args

import pytest
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.types import Command
from pydantic import BaseModel
from sqlalchemy import select, text

from agents.graph import build_graph, checkpointer_dsn
from agents.shared import ComponentStatus, ContractAction
from agents.stages import gold
from agents.stages.gold.schemas import ArchitectureChangeType, GoldEntityType
from agents.state import initial_state
from agents.template import load_json_response
from app.config import settings
from db.models import BronzeDocument, GoldAlias, GoldEvolution, SilverChunk, SilverClarification, SilverDocument
from db.session import async_session_factory
from llm import router

from agents.stages.gold.testing.conftest import record_result
from agents.stages.gold.testing.retrieval import answer_question, embed_question, top_k_gold_evolution

pytestmark = pytest.mark.anyio

SOURCE_COMPONENT = "order_fulfillment_platform.en.vtt"
GOLDEN_SET_DIR = Path(__file__).parent / "golden_set"
OUTPUT_DIR = Path(__file__).parent / "output"

# Each step name pairs with its ingestion date. Steps 06-10 continue the SAME
# order_fulfillment_platform.en.vtt history one month at a time (2026-06-15 to 2026-10-15),
# doubling this suite from 5 to 10 steps. They introduce a second, independent thread — Refund
# Service / Refund Issued — deliberately walking the exact same full ContractAction lifecycle
# Order Events already walks in steps 01-05 (new, forward-update, break-change, deprecated,
# removed; see `_REFUND_ISSUED_LIFECYCLE` below), instead of just repeating steps 01-05's own
# entities. This gives the suite twice the real Gold history to retrieve against, without
# touching Order Events or Legacy Order Monolith's already-completed arcs from steps 01-05.
STEPS: list[tuple[str, date]] = [
    ("01_order_service_launch", date(2026, 1, 15)),
    ("02_payment_service_added", date(2026, 2, 15)),
    ("03_inventory_service_and_breaking_change", date(2026, 3, 15)),
    ("04_notification_service_and_deprecation", date(2026, 4, 15)),
    ("05_legacy_monolith_removed", date(2026, 5, 15)),
    ("06_refund_service_added", date(2026, 6, 15)),
    ("07_refund_issued_forward_update", date(2026, 7, 15)),
    ("08_fraud_check_service_and_breaking_change", date(2026, 8, 15)),
    ("09_refund_processed_and_deprecation", date(2026, 9, 15)),
    ("10_refund_issued_removed", date(2026, 10, 15)),
]

# This set every valid Gold operation value, except "unknown".
_CLOSED_OPERATIONS = (
    set(get_args(ComponentStatus)) | set(get_args(ContractAction)) | set(get_args(ArchitectureChangeType))
) - {"unknown"}
_ENTITY_TYPES = set(get_args(GoldEntityType))

# To test following components
_KNOWN_COMPONENTS = {
    "Order Service",
    "Payment Service",
    "Inventory Service",
    "Notification Service",
    "Legacy Order Monolith",
    "Refund Service",
    "Fraud Check Service",
}

_ORDER_EVENTS_LIFECYCLE = ["new", "forward-update", "break-change", "deprecated", "removed"]
# Refund Issued (steps 06-10) walks the exact same ContractAction lifecycle Order Events walks
# in steps 01-05 — a second, independent proof that hash-compare-then-bump versioning holds
# across a real, growing history, not a coincidence specific to one entity.
_REFUND_ISSUED_LIFECYCLE = ["new", "forward-update", "break-change", "deprecated", "removed"]
_FALLBACK_ANSWER = "No special case here; proceed with the default."
_MAX_RESUME_ROUNDS = 5

# TRUNCATE takes a table name, not a WHERE clause. This list names every
# table this suite writes to, read from each model's own __tablename__ so it
# stays in sync with db/models.py.
_TABLES_TO_TRUNCATE = [
    GoldAlias.__tablename__,
    GoldEvolution.__tablename__,
    SilverChunk.__tablename__,
    SilverClarification.__tablename__,
    SilverDocument.__tablename__,
    BronzeDocument.__tablename__,
]


class SummaryJudgment(BaseModel):
    passes: bool
    reason: str


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


# --- Bronze setup and cleanup ---------------------------------------------------


async def _insert_bronze(ingestion_date: date, content: str) -> None:
    """Insert one bronze_documents row for this step's transcript."""
    async with async_session_factory() as session:
        session.add(
            BronzeDocument(
                ingestion_date=ingestion_date,
                source_component=SOURCE_COMPONENT,
                content=content,
                embedding=[0.0] * settings.embedding_dim,
            )
        )
        await session.commit()


async def _cleanup_all() -> None:
    """Truncate every table this suite writes to."""
    async with async_session_factory() as session:
        await session.execute(text(f"TRUNCATE {', '.join(_TABLES_TO_TRUNCATE)}"))
        await session.commit()


# --- Graph execution --------------------------------------------------------------


async def _run_to_completion(thread_id: str, payload) -> dict:
    """Run the graph to completion, and answer every interrupt along the way."""
    async with AsyncPostgresSaver.from_conn_string(checkpointer_dsn()) as saver:
        await saver.setup()
        graph = build_graph(saver)
        config = {"configurable": {"thread_id": thread_id}}
        result = await graph.ainvoke(payload, config=config)
        return await _resume_until_done(graph, config, result, thread_id)


async def _resume_until_done(graph, config: dict, result: dict, thread_id: str) -> dict:
    """Answer every interrupt in `result` with the fallback answer, and resume.

    Fail the test if the graph pauses more than _MAX_RESUME_ROUNDS times.
    """
    rounds = 0
    while "__interrupt__" in result:
        rounds += 1
        assert rounds <= _MAX_RESUME_ROUNDS, f"too many interrupt rounds for {thread_id!r}: {result}"
        pending = result["__interrupt__"][0].value["pending_questions"]
        answers = {question: _FALLBACK_ANSWER for question in pending}
        result = await graph.ainvoke(Command(resume=answers), config=config)
    return result


# --- Per-step structural checks (expected_gold_facts.json) ------------------------


async def _assert_expected_gold_facts(ingestion_date: date, expected_facts: list[dict]) -> list[str]:
    """Compare gold_evolution rows against one step's expected_gold_facts.json.
    Return a list of violation messages. An empty list means every fact matched.
    """
    rows = await _fetch_gold_rows(ingestion_date)
    by_key = {(row.entity_type, row.canonical_name.lower()): row for row in rows}

    violations: list[str] = []
    for fact in expected_facts:
        violations.extend(_check_one_fact(fact, by_key))
    return violations


async def _fetch_gold_rows(ingestion_date: date) -> list[GoldEvolution]:
    """Fetch every gold_evolution row this suite wrote on one ingestion date."""
    async with async_session_factory() as session:
        rows = (
            await session.execute(
                select(GoldEvolution).where(
                    GoldEvolution.source_component == SOURCE_COMPONENT,
                    GoldEvolution.ingestion_date == ingestion_date,
                )
            )
        ).scalars().all()
    return rows


def _check_one_fact(fact: dict, by_key: dict) -> list[str]:
    """Compare one expected fact against its matching gold_evolution row."""
    key = (fact["entity_type"], fact["canonical_name"].lower())
    row = by_key.get(key)
    if row is None:
        return [f"missing gold_evolution row for {fact}"]

    violations: list[str] = []
    if row.operation != fact["operation"]:
        violations.append(
            f"{fact['canonical_name']} ({fact['entity_type']}): expected operation "
            f"{fact['operation']!r}, got {row.operation!r}"
        )
    if "version" in fact and row.version != fact["version"]:
        violations.append(
            f"{fact['canonical_name']} ({fact['entity_type']}): expected version "
            f"{fact['version']}, got {row.version}"
        )
    return violations


# --- Per-step QA checks (qa.json) --------------------------------------------------


async def _run_qa(qa_items: list[dict], step_name: str) -> list[dict]:
    """Run every QA question for one step, and check each answer.

    For each question: retrieve the top-k Gold rows, generate an answer, and
    compare both against qa.json's expected values. Write one record per
    question to output/<step_name>/. Return every record.
    """
    results: list[dict] = []
    async with async_session_factory() as session:
        for i, item in enumerate(qa_items):
            print(f"  [qa {i + 1}/{len(qa_items)}] {item['question']!r}", flush=True)
            record = await _score_qa_item(session, item)
            results.append(record)
            _write_qa_record(step_name, i, record)
            _assert_qa_record(step_name, i, item, record)
    return results


async def _score_qa_item(session, item: dict) -> dict:
    """Retrieve Gold rows for one question, generate an answer, and score both.

    `mode="hybrid"` matches what `scripts/chat_gold.py` and `/v1/frontend/chat` actually use in
    production (see `.tmp/advanced_techniques.md` §1) — this suite tests the retrieval strategy
    real traffic gets, not a stale `mode="vector"` snapshot from before hybrid search existed."""
    vector = await embed_question(item["question"])
    rows = await top_k_gold_evolution(
        session, vector, k=8, source_component=SOURCE_COMPONENT, mode="hybrid", question_text=item["question"]
    )
    answer = await answer_question(item["question"], rows)
    expected = item["expected_entity"]

    return {
        "question": item["question"],
        "expected_entity": expected,
        "entity_hit": _matches_entity(rows, expected),
        "operation_hit": _matches_entity_and_operation(rows, expected),
        "keyword_hit": _contains_any_keyword(answer, item["expected_answer_keywords"]),
        "retrieved": _summarize_rows(rows),
        "answer": answer,
    }


def _matches_entity(rows, expected: dict) -> bool:
    """Check whether any retrieved row matches the expected entity.

    This check gates the test. It proves top-k retrieval found the right
    entity. That is the exact capability this suite exists to prove.
    """
    return any(
        row.entity_type == expected["entity_type"]
        and row.canonical_name.lower() == expected["canonical_name"].lower()
        for row in rows
    )


def _matches_entity_and_operation(rows, expected: dict) -> bool:
    """Check whether any retrieved row also matches the expected operation.

    This check is informational only. It does not gate the test. A wrong
    operation value comes from extract_gold_facts's own LLM judgment call, not
    from retrieval. agents.shared.ContractAction's own docstring already
    documents that call as having no mechanical check. _assert_expected_gold_
    facts already gates on the same wrong value, with a clearer message.
    Gating here too would only fail twice for one root cause.
    """
    return any(
        row.entity_type == expected["entity_type"]
        and row.canonical_name.lower() == expected["canonical_name"].lower()
        and row.operation == expected["operation"]
        for row in rows
    )


def _contains_any_keyword(answer: str, keywords: list[str]) -> bool:
    """Check whether the answer text contains at least one expected keyword."""
    answer_lower = answer.lower()
    return any(keyword.lower() in answer_lower for keyword in keywords)


def _summarize_rows(rows) -> list[dict]:
    """Build a small, JSON-safe summary of the retrieved Gold rows."""
    return [
        {
            "entity_type": row.entity_type,
            "canonical_name": row.canonical_name,
            "version": row.version,
            "operation": row.operation,
        }
        for row in rows
    ]


def _write_qa_record(step_name: str, index: int, record: dict) -> None:
    """Write one QA record to output/<step_name>/<index>_answer.json."""
    step_output_dir = OUTPUT_DIR / step_name
    step_output_dir.mkdir(parents=True, exist_ok=True)
    (step_output_dir / f"{index:02d}_answer.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _assert_qa_record(step_name: str, index: int, item: dict, record: dict) -> None:
    """Fail the test if this QA record misses either gating check."""
    assert record["entity_hit"], (
        f"{step_name} qa[{index}] {item['question']!r}: top-k retrieval missed "
        f"{record['expected_entity']}; got {record['retrieved']}"
    )
    assert record["keyword_hit"], (
        f"{step_name} qa[{index}] {item['question']!r}: generated answer missing any of "
        f"{item['expected_answer_keywords']}: {record['answer']!r}"
    )


# --- Suite-level checklist (checklist.txt) -----------------------------------------


async def _critique_summary(summary_answer: str) -> dict:
    """Ask a second LLM to judge one free-form summary answer.

    This check is informational only. It never gates the test.
    """
    messages = [{"role": "user", "content": _summary_judgment_prompt(summary_answer)}]
    response = await router.complete(
        messages, response_format=SummaryJudgment, temperature=0, reasoning_effort="none"
    )
    return SummaryJudgment.model_validate(load_json_response(response.choices[0].message.content)).model_dump()


def _summary_judgment_prompt(summary_answer: str) -> str:
    """Build the judgment prompt text for one summary answer."""
    expected_story = (
        "launched with a new Order Service alongside a Legacy Order Monolith kept running "
        "unchanged; added Payment Service, with Order Events gaining a backward-compatible "
        "optional field; added Inventory Service alongside a breaking change to Order Events' "
        "buyer_id field; added Notification Service and deprecated Order Events; then retired "
        "Legacy Order Monolith and removed Order Events entirely; separately, added a new "
        "Refund Service publishing a new Refund Issued contract; gave Refund Issued a "
        "backward-compatible optional 'reason' field; added a new Fraud Check Service alongside "
        "a breaking change to Refund Issued's amount_refunded field; introduced a consolidated "
        "Refund Processed contract and deprecated Refund Issued; then removed Refund Issued "
        "entirely, fully replaced by Refund Processed"
    )
    return (
        f"Judge whether this summary of an architecture's evolution accurately reflects a "
        f"system that: {expected_story}. This is informational only, not a gate — report your "
        f"honest read, including anything the summary gets wrong or omits.\n\n"
        f"Summary:\n{summary_answer}"
    )


async def _check_contract_lifecycle(canonical_name: str, expected_lifecycle: list[str]) -> tuple[list[str], list[str]]:
    """Checklist items 1 and 2: one data contract must walk its full, exact ContractAction
    lifecycle, in version order. Shared by Order Events (steps 01-05) and Refund Issued (steps
    06-10) — the same check, run twice, against two independent entities, is a stronger proof
    that hash-compare-then-bump versioning holds in general than running it against only one
    entity ever would be.

    Return the list of violations, and the actual lifecycle found.
    """
    async with async_session_factory() as session:
        rows = (
            await session.execute(
                select(GoldEvolution)
                .where(
                    GoldEvolution.entity_type == "data_contract",
                    GoldEvolution.canonical_name == canonical_name,
                    GoldEvolution.source_component == SOURCE_COMPONENT,
                )
                .order_by(GoldEvolution.version)
            )
        ).scalars().all()

    actual_lifecycle = [row.operation for row in rows]
    if actual_lifecycle == expected_lifecycle:
        return [], actual_lifecycle
    violation = f"{canonical_name} lifecycle mismatch: expected {expected_lifecycle}, got {actual_lifecycle}"
    return [violation], actual_lifecycle


async def _check_legacy_monolith_retrieval() -> tuple[list[str], str]:
    """Checklist item 3: retrieval must surface the monolith's removed row.

    Return the list of violations, and the generated answer text.
    """
    print("  [checklist 3/6] legacy monolith question", flush=True)
    question = "What happened to the legacy monolith?"
    async with async_session_factory() as session:
        vector = await embed_question(question)
        rows = await top_k_gold_evolution(
            session, vector, k=8, source_component=SOURCE_COMPONENT, mode="hybrid", question_text=question
        )

    removed_hit = any(
        row.entity_type == "component"
        and row.canonical_name.lower() == "legacy order monolith"
        and row.operation == "removed"
        for row in rows
    )
    if not removed_hit:
        violation = (
            "top-k for the legacy monolith question missed its removed row: "
            f"{[(row.canonical_name, row.operation) for row in rows]}"
        )
        return [violation], ""

    answer = await answer_question(question, rows)
    if any(keyword in answer.lower() for keyword in ("removed", "retired", "decommissioned")):
        return [], answer
    violation = f"generated answer for the legacy monolith question lacks an expected keyword: {answer!r}"
    return [violation], answer


async def _check_closed_vocabulary() -> list[str]:
    """Checklist item 4: every gold_evolution row must use a closed vocabulary.

    No row may use an entity_type or operation outside the closed sets. In
    particular, no row may use operation == "unknown".
    """
    async with async_session_factory() as session:
        rows = (
            await session.execute(
                select(GoldEvolution.entity_type, GoldEvolution.operation).where(
                    GoldEvolution.source_component == SOURCE_COMPONENT
                )
            )
        ).all()

    bad_entity_types = {entity_type for entity_type, _ in rows if entity_type not in _ENTITY_TYPES}
    bad_operations = {operation for _, operation in rows if operation not in _CLOSED_OPERATIONS}

    violations: list[str] = []
    if bad_entity_types:
        violations.append(f"gold_evolution rows with entity_type outside the closed set: {bad_entity_types}")
    if bad_operations:
        violations.append(f"gold_evolution rows with operation outside the closed vocabulary: {bad_operations}")
    return violations


async def _check_current_gold_state() -> list[str]:
    """Checklist item 5: current_gold_state must report the right live components.

    Every component except the retired monolith must still be present. The
    monolith's latest state must be "removed".
    """
    async with async_session_factory() as session:
        current_rows = await gold.current_gold_state(session, entity_type="component")

    ours = {row.canonical_name: row for row in current_rows if row.canonical_name in _KNOWN_COMPONENTS}
    expected_present = {
        "Order Service",
        "Payment Service",
        "Inventory Service",
        "Notification Service",
        "Refund Service",
        "Fraud Check Service",
    }
    present = {name for name, row in ours.items() if row.operation != "removed"}

    violations: list[str] = []
    if present != expected_present:
        violations.append(f"current_gold_state(component) mismatch: expected {expected_present}, got {present}")

    monolith = ours.get("Legacy Order Monolith")
    if monolith is not None and monolith.operation != "removed":
        violations.append("Legacy Order Monolith's latest gold_evolution state is not 'removed'")
    return violations


async def _check_summary_judgment() -> tuple[str, dict]:
    """Checklist item 6: judge a free-form summary of the whole history.

    This check is informational only. It never gates the test. Return the
    generated summary answer, and the judgment result.
    """
    print("  [checklist 6/6] summary question + judgment", flush=True)
    question = "Summarize how this architecture evolved from start to finish."
    async with async_session_factory() as session:
        vector = await embed_question(question)
        # k=15, not the original 10: doubling the suite to 10 steps also roughly doubled the
        # number of distinct entities in this history (~14, across both the Order Events and
        # Refund Issued threads plus every component and the architecture entity). A narrower
        # k, even with dedup, would silently drop one of the two threads from this summary.
        rows = await top_k_gold_evolution(
            session, vector, k=15, source_component=SOURCE_COMPONENT, mode="hybrid", question_text=question
        )

    answer = await answer_question(question, rows)
    judgment = await _critique_summary(answer)
    return answer, judgment


def _write_checklist_outputs(summary_answer: str, judgment: dict) -> None:
    """Write the checklist's summary answer and judgment result to disk."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "checklist_summary_answer.txt").write_text(summary_answer, encoding="utf-8")
    (OUTPUT_DIR / "checklist_judgment.json").write_text(
        json.dumps(judgment, indent=2, ensure_ascii=False), encoding="utf-8"
    )


async def _run_checklist_deterministic() -> dict:
    """Run every item in checklist.txt, and collect every violation.

    Items 1 to 5 are deterministic. They gate the test. Item 6 is a judgment
    call. It is informational only.
    """
    order_events_violations, order_events_lifecycle = await _check_contract_lifecycle(
        "Order Events", _ORDER_EVENTS_LIFECYCLE
    )
    refund_issued_violations, refund_issued_lifecycle = await _check_contract_lifecycle(
        "Refund Issued", _REFUND_ISSUED_LIFECYCLE
    )
    monolith_violations, monolith_answer = await _check_legacy_monolith_retrieval()
    vocabulary_violations = await _check_closed_vocabulary()
    state_violations = await _check_current_gold_state()
    summary_answer, judgment = await _check_summary_judgment()
    _write_checklist_outputs(summary_answer, judgment)

    violations = [
        *order_events_violations,
        *refund_issued_violations,
        *monolith_violations,
        *vocabulary_violations,
        *state_violations,
    ]
    return {
        "violations": violations,
        "details": {
            "order_events_lifecycle": order_events_lifecycle,
            "refund_issued_lifecycle": refund_issued_lifecycle,
            "monolith_answer": monolith_answer,
            "summary_answer": summary_answer,
            "judgment": judgment,
        },
    }


# --- Debug output -------------------------------------------------------------------


def _write_extraction_debug(step_name: str, result: dict) -> None:
    """Write the graph's raw extraction output to disk, before any check runs.

    _cleanup_all() truncates the real gold_evolution and gold_aliases tables
    after every run, pass or fail. result.json only records what reached those
    tables, not what the LLM produced before that. Without this file, a
    failure like "expected data_contract row missing" gives no way to tell
    where the fact was lost: mentioned_data_contracts, extract_gold_facts, or
    somewhere else.
    """
    step_output_dir = OUTPUT_DIR / step_name
    step_output_dir.mkdir(parents=True, exist_ok=True)
    debug = {
        "mentioned_components": result.get("mentioned_components"),
        "mentioned_data_contracts": result.get("mentioned_data_contracts"),
        "gold_extractions": result.get("gold_extractions"),
        "documents": result.get("documents"),
    }
    (step_output_dir / "extraction_debug.json").write_text(
        json.dumps(debug, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )


# --- Test entry point -----------------------------------------------------------


async def _run_one_step(step_index: int, step_name: str, ingestion_date: date) -> None:
    """Run one golden-set step end to end, and check its own facts."""
    print(f"\n=== [{step_index}/{len(STEPS)}] {step_name} ({ingestion_date}) — running the graph ===", flush=True)
    case_dir = GOLDEN_SET_DIR / step_name
    transcript = (case_dir / "transcript.txt").read_text(encoding="utf-8")
    expected_facts = json.loads((case_dir / "expected_gold_facts.json").read_text(encoding="utf-8"))
    qa_items = json.loads((case_dir / "qa.json").read_text(encoding="utf-8"))

    await _insert_bronze(ingestion_date, transcript)

    ingestion_date_str = ingestion_date.strftime("%Y%m%d")
    result = await _run_to_completion(f"gold-evolution-{step_name}", initial_state(ingestion_date_str))
    _write_extraction_debug(step_name, result)
    assert "__interrupt__" not in result

    print(f"=== [{step_index}/{len(STEPS)}] {step_name} — graph done, checking facts + running QA ===", flush=True)
    violations = await _assert_expected_gold_facts(ingestion_date, expected_facts)
    qa_results = await _run_qa(qa_items, step_name)

    record_result(step_name, not violations, violations, {"qa": qa_results})
    assert not violations, f"{step_name}: structural Gold facts check failed: {violations}"


async def test_gold_arch_evolution_five_steps() -> None:
    """Run all 5 steps in order, then run the suite-level checklist."""
    await _cleanup_all()  # A previous failed run may have left rows behind.
    try:
        for step_index, (step_name, ingestion_date) in enumerate(STEPS, start=1):
            await _run_one_step(step_index, step_name, ingestion_date)

        print(f"\n=== checklist.txt — suite-level deterministic checks over all {len(STEPS)} steps ===", flush=True)
        checklist_outcome = await _run_checklist_deterministic()
        record_result(
            "checklist_general",
            not checklist_outcome["violations"],
            checklist_outcome["violations"],
            checklist_outcome["details"],
        )
        assert not checklist_outcome["violations"], f"checklist.txt: {checklist_outcome['violations']}"
    finally:
        await _cleanup_all()
