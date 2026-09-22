import json
import shutil
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import litellm
import pytest
from fastapi import HTTPException
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from sqlalchemy import delete, select

from agents.graph import (
    _drop_new_contract_version_questions,
    _ensure_schema_questions,
    build_graph,
    checkpointer_dsn,
    synthesize_document,
)
from agents.stages.adr_generation.service import SUGGEST_INFO_CORRECTION_MESSAGE
from agents.state import initial_state
from app.config import settings
from app.routers.frontend import _resume_graph
from db.models import BronzeDocument, GoldAlias, GoldEvolution, SilverChunk, SilverClarification, SilverDocument
from db.session import async_session_factory

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _fake_response(content: str) -> SimpleNamespace:
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice], model="fake")


_DEFAULT_QUESTION = {
    "id": "placeholder.question",
    "scope": "component",
    "target": "placeholder",
    "requirement": "placeholder",
    "question": "placeholder question?",
}

# This is the harmless default for extract_gold_facts's one LLM call. Nothing is extracted.
# The architecture is unchanged. _make_fake_acompletion uses this as its own default. Any
# test below that fakes its own acompletion, but does not test Gold extraction, shares it
# too.
_DEFAULT_GOLD_EXTRACTION = {
    "components": [],
    "contracts": [],
    "architecture_change": "unchanged",
    "architecture_narrative": "No architecture change extracted for this test.",
    "mermaid_diagram": "",
}


def _make_fake_acompletion(
    classification: dict,
    synthesis_queue: list[str],
    critique_queue: list[dict],
    generated_questions: list[dict] | None = None,
    mentioned_components: list[dict] | None = None,
    mentioned_data_contracts: list[dict] | None = None,
    data_contract_questions: list[dict] | None = None,
    gold_extraction: dict | None = None,
    completeness_score: int = 100,
    unresolved_points: list[str] | None = None,
):
    """This function dispatches based on which prompt was sent: architecture questions,
    data-contract questions, classify, synthesize, or critic. Each question-generation stage
    and `classify_questions` send their whole prompt as a single `user` message. For the
    first two, that message is loaded verbatim from `prompts/architecture_questions/
    combined.jinja` or `data_contract_questions.jinja`. synthesize and critic still use a
    `system` plus `user` pair. Either way, check `messages[0]["content"]`. `synthesis_queue`
    and `critique_queue` are consumed in call order. Each entry matches one
    synthesize_document or critic_document call, in the order the graph actually makes them:
    first pass, then one more per redraft.

    `classification`'s entries are matched back to `generated_questions` by `id`. See
    `agents.graph.classify_questions`. So a test that overrides one must override the other
    consistently. The default single placeholder question/classification pair is enough for
    tests that don't care about specific question text or ids. Leaving
    `mentioned_data_contracts` at its default (empty) means the data-contract stage skips its
    LLM call entirely. This is `generate_data_contract_questions_for_batch`'s own short
    circuit. So most tests below never need to fake that prompt at all.

    `gold_extraction` fakes `extract_gold_facts`'s one LLM call per source, in
    `agents/graph.py`. This call now runs unconditionally for every source that reaches
    `boss_verdicts[source] == "ok"`. So every pre-existing test below also exercises this
    node. Each test uses the harmless default, nothing extracted and architecture unchanged,
    unless it overrides it. This works the same way `mentioned_data_contracts` defaulting to
    empty means most tests never bother faking the data-contract stage either. See
    `test_gold_extraction_persists_and_skips_unknown_status` for the one test that overrides
    it to exercise real extraction and persistence."""
    questions = generated_questions if generated_questions is not None else [_DEFAULT_QUESTION]
    components = mentioned_components if mentioned_components is not None else []
    contracts = mentioned_data_contracts if mentioned_data_contracts is not None else []
    contract_questions = data_contract_questions if data_contract_questions is not None else []
    gold_result = gold_extraction if gold_extraction is not None else _DEFAULT_GOLD_EXTRACTION

    async def fake_acompletion(*, model, api_key, messages, **kwargs):
        content_in = messages[0]["content"]
        if "Senior Software/Data Architect" in content_in:
            content = json.dumps(
                {"mentioned_components": components, "mentioned_data_contracts": contracts, "questions": questions}
            )
        elif "Senior Data Governance Requirements Analyst" in content_in:
            content = json.dumps({"questions": contract_questions})
        elif "Classify each question" in content_in:
            content = json.dumps(classification)
        elif "extracting structured, versionable facts" in content_in:
            content = json.dumps(gold_result)
        elif "Architecture Decision Record" in content_in:
            content = synthesis_queue.pop(0)
        elif "Review the drafted document" in content_in:
            content = json.dumps(
                {
                    "claims": critique_queue.pop(0),
                    "completeness_score": completeness_score,
                    "unresolved_points": unresolved_points if unresolved_points is not None else [],
                }
            )
        else:
            raise AssertionError(f"unexpected prompt: {content_in[:80]!r}")
        return _fake_response(content)

    return fake_acompletion


async def _insert_bronze(ingestion_date: date, source_component: str, contents: list[str]) -> None:
    async with async_session_factory() as session:
        for content in contents:
            session.add(
                BronzeDocument(
                    ingestion_date=ingestion_date,
                    source_component=source_component,
                    content=content,
                    embedding=[0.0] * settings.embedding_dim,
                )
            )
        await session.commit()


async def _cleanup_date(ingestion_date: date) -> None:
    """This also cleans up any `gold_evolution` and `gold_aliases` rows this run wrote.
    `extract_gold_facts` now runs unconditionally for every clean-passing source. See
    `_make_fake_acompletion`'s `gold_result` default. So every test in this file writes at
    least an `architecture:<source>` row, unless it explicitly skips Gold. `gold_aliases`
    has no `ingestion_date` column of its own. It is not versioned. See `gold_process.md`
    §3. So we find its rows through the `(entity_type, entity_id)` pairs `gold_evolution`
    just gave us. We delete those rows before the `gold_evolution` rows themselves, so
    nothing is left to look up."""
    async with async_session_factory() as session:
        gold_entities = (
            await session.execute(
                select(GoldEvolution.entity_type, GoldEvolution.entity_id).where(
                    GoldEvolution.ingestion_date == ingestion_date
                )
            )
        ).all()
        for entity_type, entity_id in gold_entities:
            await session.execute(
                delete(GoldAlias).where(GoldAlias.entity_type == entity_type, GoldAlias.entity_id == entity_id)
            )
        await session.execute(delete(GoldEvolution).where(GoldEvolution.ingestion_date == ingestion_date))
        await session.execute(delete(SilverChunk).where(SilverChunk.ingestion_date == ingestion_date))
        await session.execute(
            delete(SilverClarification).where(SilverClarification.ingestion_date == ingestion_date)
        )
        await session.execute(delete(SilverDocument).where(SilverDocument.ingestion_date == ingestion_date))
        await session.execute(delete(BronzeDocument).where(BronzeDocument.ingestion_date == ingestion_date))
        await session.commit()

    questions_dir = (
        Path(settings.output_dir) / f"ingestion_date={ingestion_date.strftime('%Y%m%d')}"
    )
    if questions_dir.is_dir():
        shutil.rmtree(questions_dir)


async def _run(thread_id: str, payload) -> dict:
    """This opens a brand-new `AsyncPostgresSaver` connection and compiles a fresh graph
    for every call. So every multi-call test below already proves the checkpoint survives
    across separate connections and processes. It is not just in-memory state within one
    Python object."""
    async with AsyncPostgresSaver.from_conn_string(checkpointer_dsn()) as saver:
        await saver.setup()
        graph = build_graph(saver)
        config = {"configurable": {"thread_id": thread_id}}
        return await graph.ainvoke(payload, config=config)


DATE_GENERATED_QUESTIONS = date(2026, 6, 6)
DATE_CLEAN = date(2026, 6, 1)
DATE_CLASSIFY_INTERRUPT = date(2026, 6, 2)
DATE_CONTRADICTION = date(2026, 6, 3)
DATE_LOW_SEVERITY = date(2026, 6, 4)
DATE_BOUNDED_RETRY = date(2026, 6, 5)
DATE_DATA_CONTRACT_QUESTIONS = date(2026, 6, 7)
DATE_MENTIONED_NAMES = date(2026, 6, 8)
DATE_GOLD_EXTRACTION = date(2026, 6, 9)
DATE_MERMAID_DOWNGRADE = date(2026, 6, 10)
DATE_GOLD_DISCOVERS_NEW_COMPONENT = date(2026, 6, 11)
DATE_DUPLICATE_MATERIAL_CLAIMS = date(2026, 6, 12)
DATE_CRITIC_SCORE = date(2026, 6, 13)
DATE_IRRELEVANT_MARKER = date(2026, 6, 14)
DATE_CLASSIFY_SEMANTIC_DUPLICATE = date(2026, 6, 15)
DATE_NEW_CONTRACT_VERSION = date(2026, 6, 16)
DATE_GOLD_CONTRACT_PRODUCER_CONSUMER_IDS = date(2026, 6, 17)
DATE_RESUME_TENANT_GUARDRAIL = date(2026, 6, 18)


async def test_data_contract_stage_questions_are_appended_to_architecture_stage_ones(monkeypatch):
    """The two question-generation stages both feed `generated_questions`.
    `generate_data_contract_questions` must append onto what `generate_architecture_questions`
    already drafted. It must not replace it. So `classify_questions` sees the union of both."""
    await _insert_bronze(
        DATE_DATA_CONTRACT_QUESTIONS, "meeting_g.en.vtt", ["Checkout publishes checkout-completed."]
    )
    try:
        architecture_question = {
            "id": "component.checkout_service.status",
            "scope": "component",
            "target": "checkout service",
            "requirement": "status",
            "question": "Is the checkout service new or existing?",
        }
        contract_question = {
            "id": "contract.checkout_completed.schema",
            "scope": "data_contract",
            "target": "checkout-completed",
            "requirement": "schema",
            "question": "What fields does checkout-completed carry?",
        }
        seen_ids: list[str] = []

        fake = _make_fake_acompletion(
            classification={"classifications": []},
            synthesis_queue=["# ADR — Checkout\n\nUnchanged."],
            critique_queue=[[]],
            generated_questions=[architecture_question],
            mentioned_components=[{"name": "checkout service", "status": "unknown"}],
            mentioned_data_contracts=[
                {
                    "name": "checkout-completed",
                    "producer": "checkout service",
                    "consumer": "unknown",
                    "action": "unknown",
                }
            ],
            data_contract_questions=[contract_question],
        )

        async def tracking_fake(*, model, api_key, messages, **kwargs):
            if "Classify each question" in messages[0]["content"]:
                seen_ids.extend(
                    line.split("id: ", 1)[1]
                    for line in messages[0]["content"].splitlines()
                    if line.strip().startswith("- id:")
                )
            return await fake(model=model, api_key=api_key, messages=messages, **kwargs)

        monkeypatch.setattr(litellm, "acompletion", tracking_fake)

        result = await _run("data-contract-questions-1", initial_state("20260607"))
        assert "__interrupt__" not in result

        assert seen_ids == ["component.checkout_service.status", "contract.checkout_completed.schema"]
    finally:
        await _cleanup_date(DATE_DATA_CONTRACT_QUESTIONS)


async def test_new_contract_version_question_is_dropped_before_classification(monkeypatch):
    """Regression test for `agents.graph._drop_new_contract_version_questions`. A `new`
    contract's version is always `1.0.0` by convention — see `prompts/adr_generation/
    generator.jinja`'s own rule, which `prompts/data_contract_questions/questions.jinja`'s
    PHASE 3 now states too. A human must never be asked for it, even when the
    data-contract-question stage drafts that question anyway despite the prompt's own
    instruction. This question must never even reach `classify_questions`."""
    await _insert_bronze(
        DATE_NEW_CONTRACT_VERSION, "meeting_e.en.vtt", ["We're launching a new OrderCreated contract."]
    )
    try:
        version_question = {
            "id": "contract.ordercreated.version",
            "scope": "data_contract",
            "target": "OrderCreated",
            "requirement": "version",
            "question": "What is the initial semantic version of the OrderCreated contract?",
        }
        schema_question = {
            "id": "contract.ordercreated.schema",
            "scope": "data_contract",
            "target": "OrderCreated",
            "requirement": "schema",
            "question": "What fields does OrderCreated carry?",
        }
        seen_ids: list[str] = []

        fake = _make_fake_acompletion(
            classification={"classifications": []},
            synthesis_queue=["# ADR — Orders\n\nOrderCreated introduced."],
            critique_queue=[[]],
            mentioned_components=[{"name": "OrderService", "status": "new"}],
            mentioned_data_contracts=[
                {
                    "name": "OrderCreated",
                    "producer": "OrderService",
                    "consumer": "FulfillmentService",
                    "action": "new",
                }
            ],
            data_contract_questions=[version_question, schema_question],
        )

        async def tracking_fake(*, model, api_key, messages, **kwargs):
            if "Classify each question" in messages[0]["content"]:
                seen_ids.extend(
                    line.split("id: ", 1)[1]
                    for line in messages[0]["content"].splitlines()
                    if line.strip().startswith("- id:")
                )
            return await fake(model=model, api_key=api_key, messages=messages, **kwargs)

        monkeypatch.setattr(litellm, "acompletion", tracking_fake)

        result = await _run("new-contract-version-1", initial_state("20260616"))
        assert "__interrupt__" not in result

        assert "contract.ordercreated.version" not in seen_ids
        assert "contract.ordercreated.schema" in seen_ids
    finally:
        await _cleanup_date(DATE_NEW_CONTRACT_VERSION)


async def test_generate_questions_output_is_what_classify_questions_actually_sees(monkeypatch):
    """generate_architecture_questions drafts per-component questions from the transcript
    (prompts/architecture_questions/combined.jinja); classify_questions must
    receive exactly that list, not some other/fixed one."""
    await _insert_bronze(
        DATE_GENERATED_QUESTIONS, "meeting_f.en.vtt", ["The checkout service was discussed."]
    )
    try:
        per_component_questions = [
            {
                "id": "component.checkout_service.status",
                "scope": "component",
                "target": "checkout service",
                "requirement": "status",
                "question": "Is the checkout service new, an evolution of something already known, or unchanged?",
            },
            {
                "id": "component.checkout_service.purpose",
                "scope": "component",
                "target": "checkout service",
                "requirement": "purpose",
                "question": "What does the checkout service do?",
            },
        ]
        mentioned_components = [{"name": "checkout service", "status": "unknown"}]
        seen_questions_blocks: list[str] = []

        async def fake_acompletion(*, model, api_key, messages, **kwargs):
            content_in = messages[0]["content"]
            if "Senior Software/Data Architect" in content_in:
                assert "The checkout service was discussed." in content_in  # got the real transcript
                assert "# Architecture Description / Evolution" in content_in  # got the real template
                return _fake_response(
                    json.dumps(
                        {
                            "mentioned_components": mentioned_components,
                            "mentioned_data_contracts": [],
                            "questions": per_component_questions,
                        }
                    )
                )
            if "Classify each question" in content_in:
                seen_questions_blocks.append(messages[0]["content"])
                return _fake_response(json.dumps({"classifications": []}))
            if "Architecture Decision Record" in content_in:
                return _fake_response("# ADR — Checkout Service\n\nCheckout: unchanged.")
            if "Review the drafted document" in content_in:
                return _fake_response(
                    json.dumps({"claims": [], "completeness_score": 100, "unresolved_points": []})
                )
            if "extracting structured, versionable facts" in content_in:
                return _fake_response(json.dumps(_DEFAULT_GOLD_EXTRACTION))
            raise AssertionError(f"unexpected prompt: {content_in[:80]!r}")

        monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

        result = await _run("generated-questions-1", initial_state("20260606"))
        assert "__interrupt__" not in result

        assert len(seen_questions_blocks) == 1
        for item in per_component_questions:
            assert item["question"] in seen_questions_blocks[0]

        # This file is output/ingestion_date=20260606/questions/meeting_f.json. Its name
        # mirrors input/transcriptions/ingestion_date=20260606/meeting_f.en.vtt's own name.
        questions_file = (
            Path(settings.output_dir) / "ingestion_date=20260606" / "questions" / "meeting_f.json"
        )
        assert questions_file.is_file()
        assert json.loads(questions_file.read_text()) == {
            "mentioned_components": mentioned_components,
            "mentioned_data_contracts": [],
            "questions": per_component_questions,
        }
    finally:
        await _cleanup_date(DATE_GENERATED_QUESTIONS)


async def test_clean_transcript_completes_with_no_interrupt_and_persists(monkeypatch):
    await _insert_bronze(DATE_CLEAN, "meeting_a.en.vtt", ["Team discussed the billing service."])
    try:
        classification = {
            "classifications": [{"id": "placeholder.question", "answer": "Billing", "status": "answered"}]
        }
        synthesized = "# Architecture Description / Evolution\n\nBilling service is new."

        monkeypatch.setattr(
            litellm,
            "acompletion",
            _make_fake_acompletion(classification, [synthesized], [[]]),
        )
        result = await _run("clean-1", initial_state("20260601"))
        assert "__interrupt__" not in result

        async with async_session_factory() as session:
            docs = (
                (await session.execute(select(SilverDocument).where(SilverDocument.ingestion_date == DATE_CLEAN)))
                .scalars()
                .all()
            )
            chunks = (
                (await session.execute(select(SilverChunk).where(SilverChunk.ingestion_date == DATE_CLEAN)))
                .scalars()
                .all()
            )
        assert len(docs) == 1
        assert docs[0].source_component == "meeting_a.en.vtt"
        assert "Billing service is new." in docs[0].content
        assert len(chunks) > 0
        assert all(c.embedding is not None for c in chunks)

        # This re-runs for the same ingestion_date. It upserts in place. No duplicates are created.
        monkeypatch.setattr(
            litellm,
            "acompletion",
            _make_fake_acompletion(classification, [synthesized], [[]]),
        )
        await _run("clean-2", initial_state("20260601"))

        async with async_session_factory() as session:
            docs_after = (
                (await session.execute(select(SilverDocument).where(SilverDocument.ingestion_date == DATE_CLEAN)))
                .scalars()
                .all()
            )
        assert len(docs_after) == 1  # upserted, not duplicated
    finally:
        await _cleanup_date(DATE_CLEAN)


async def test_critic_completeness_score_reaches_the_final_state(monkeypatch):
    """`critic_document`, in `agents/graph.py`, must show the Critic's own
    `completeness_score` and `unresolved_points`. These come from
    prompts/adr_critic/critic.jinja's COMPLETENESS SCORING section. It must put them in
    `state["adr_scores"]` and `state["adr_unresolved_points"]`, keyed by source.
    `app/routers/frontend.py::_to_response` reads these values. It uses them to show the
    reviewer a completeness percentage and the specific unresolved fields, instead of a
    silently-placeholder ADR."""
    await _insert_bronze(DATE_CRITIC_SCORE, "meeting_score.en.vtt", ["The reporting pipeline was discussed."])
    try:
        monkeypatch.setattr(
            litellm,
            "acompletion",
            _make_fake_acompletion(
                classification={"classifications": []},
                synthesis_queue=["# Architecture Description / Evolution\n\nReporting pipeline is unchanged."],
                critique_queue=[[]],
                completeness_score=55,
                unresolved_points=["reporting-events contract: version not specified"],
            ),
        )
        result = await _run("critic-score-1", initial_state("20260613"))
        assert "__interrupt__" not in result

        assert result["adr_scores"]["meeting_score.en.vtt"] == 55
        assert result["adr_unresolved_points"]["meeting_score.en.vtt"] == [
            "reporting-events contract: version not specified"
        ]
    finally:
        await _cleanup_date(DATE_CRITIC_SCORE)


async def test_write_document_grounds_batch_mentions_per_source(monkeypatch):
    """`generate_architecture_questions` drafts `mentioned_components` and
    `mentioned_data_contracts` once, over the whole batch's pooled transcript. See that
    function's docstring. `write_document` must ground each mention against its own
    source's content, using `agents.shared.mentions_grounded_in_source`, before persisting
    it on that source's `SilverDocument` row. It must not duplicate the whole batch's list
    onto every row in the batch."""
    await _insert_bronze(
        DATE_MENTIONED_NAMES, "meeting_checkout.en.vtt", ["We are introducing the checkout service."]
    )
    await _insert_bronze(
        DATE_MENTIONED_NAMES, "meeting_billing.en.vtt", ["The billing service remains unchanged."]
    )
    try:
        mentioned_components = [
            {"name": "checkout service", "status": "new"},
            {"name": "billing service", "status": "unchanged"},
        ]
        fake = _make_fake_acompletion(
            classification={"classifications": []},
            synthesis_queue=[
                "# ADR — Checkout\n\nCheckout service is new.",
                "# ADR — Billing\n\nBilling service is unchanged.",
            ],
            critique_queue=[[], []],
            mentioned_components=mentioned_components,
        )
        monkeypatch.setattr(litellm, "acompletion", fake)

        result = await _run("mentioned-names-1", initial_state("20260608"))
        assert "__interrupt__" not in result

        async with async_session_factory() as session:
            docs = (
                (
                    await session.execute(
                        select(SilverDocument).where(SilverDocument.ingestion_date == DATE_MENTIONED_NAMES)
                    )
                )
                .scalars()
                .all()
            )
        by_source = {d.source_component: d for d in docs}
        assert by_source["meeting_checkout.en.vtt"].mentioned_component_names == [
            {"name": "checkout service", "status": "new"}
        ]
        assert by_source["meeting_billing.en.vtt"].mentioned_component_names == [
            {"name": "billing service", "status": "unchanged"}
        ]
        assert by_source["meeting_checkout.en.vtt"].mentioned_data_contract_names == []
        assert by_source["meeting_billing.en.vtt"].mentioned_data_contract_names == []
    finally:
        await _cleanup_date(DATE_MENTIONED_NAMES)


async def test_classify_stage_interrupt_and_resume_persists_across_new_connection(monkeypatch):
    await _insert_bronze(DATE_CLASSIFY_INTERRUPT, "meeting_b.en.vtt", ["Someone mentioned a new service."])
    try:
        fake = _make_fake_acompletion(
            classification={
                "classifications": [
                    {"id": "component.new_service.owner", "answer": None, "status": "needs_clarification"}
                ]
            },
            synthesis_queue=["# Architecture Description / Evolution\n\nOwner: Alex."],
            critique_queue=[[]],
            generated_questions=[
                {
                    "id": "component.new_service.owner",
                    "scope": "component",
                    "target": "new service",
                    "requirement": "owner",
                    "question": "who owns this?",
                }
            ],
        )
        monkeypatch.setattr(litellm, "acompletion", fake)

        first = await _run("classify-interrupt-1", initial_state("20260602"))
        assert "__interrupt__" in first
        payload = first["__interrupt__"][0].value
        assert payload["origin"] == "classify"
        assert payload["pending_questions"] == ["who owns this?"]

        from langgraph.types import Command

        second = await _run("classify-interrupt-1", Command(resume={"who owns this?": "Alex"}))
        assert "__interrupt__" not in second

        async with async_session_factory() as session:
            audit_row = (
                (
                    await session.execute(
                        select(SilverDocument).where(SilverDocument.ingestion_date == DATE_CLASSIFY_INTERRUPT)
                    )
                )
                .scalars()
                .one()
            )
            clarification_row = (
                (
                    await session.execute(
                        select(SilverClarification).where(
                            SilverClarification.ingestion_date == DATE_CLASSIFY_INTERRUPT
                        )
                    )
                )
                .scalars()
                .one()
            )
        assert "Owner: Alex." in audit_row.content
        # Silver stores nothing on disk. The audit trail lives in silver_clarifications.
        assert clarification_row.question == "who owns this?"
        assert clarification_row.answer == "Alex"
    finally:
        await _cleanup_date(DATE_CLASSIFY_INTERRUPT)


async def test_resume_transcription_guardrail_rejects_a_thread_from_a_different_tenant(monkeypatch):
    """Guardrail test for `app.routers.frontend._resume_graph`. `thread_id` is a plain string a
    client can type, log, or guess (`f"frontend-{tenant}-{ingestion_date}-{uuid}"`, see
    `upload_transcription`) — nothing about it proves the caller actually belongs to the tenant
    it was minted for. Without `_resume_graph`'s own tenant check, a request that knew or
    guessed another tenant's thread_id could resume THAT tenant's paused clarification session
    and read its draft ADR content back in the response. This drives a real graph run to a
    genuine paused state for one tenant, then tries to resume it as a different one."""
    tenant_a, tenant_b = "guardrail-tenant-a", "guardrail-tenant-b"
    async with async_session_factory() as session:
        session.add(
            BronzeDocument(
                ingestion_date=DATE_RESUME_TENANT_GUARDRAIL,
                source_component="meeting_resume_guardrail.en.vtt",
                content="Someone mentioned a new service.",
                embedding=[0.0] * settings.embedding_dim,
                tenant=tenant_a,
            )
        )
        await session.commit()
    try:
        fake = _make_fake_acompletion(
            classification={
                "classifications": [
                    {"id": "component.new_service.owner", "answer": None, "status": "needs_clarification"}
                ]
            },
            synthesis_queue=["# ADR — placeholder"],
            critique_queue=[[]],
            generated_questions=[
                {
                    "id": "component.new_service.owner",
                    "scope": "component",
                    "target": "new service",
                    "requirement": "owner",
                    "question": "who owns this?",
                }
            ],
        )
        monkeypatch.setattr(litellm, "acompletion", fake)

        thread_id = "resume-tenant-guardrail-1"
        first = await _run(
            thread_id,
            initial_state(
                DATE_RESUME_TENANT_GUARDRAIL.strftime("%Y%m%d"), tenant=tenant_a, persist=False
            ),
        )
        assert "__interrupt__" in first

        with pytest.raises(HTTPException) as exc_info:
            await _resume_graph(thread_id, {"who owns this?": "Alex"}, expected_tenant=tenant_b)
        assert exc_info.value.status_code == 404

        # The real owning tenant can still resume it normally — the guardrail only blocks a
        # mismatched tenant, it does not break the legitimate one.
        second = await _resume_graph(thread_id, {"who owns this?": "Alex"}, expected_tenant=tenant_a)
        assert "__interrupt__" not in second
    finally:
        await _cleanup_date(DATE_RESUME_TENANT_GUARDRAIL)


async def test_classify_stage_folds_semantic_duplicate_questions_into_one_pending_question(monkeypatch):
    """Regression test for the classify stage's DUPLICATE QUESTIONS handling
    (`agents.graph._canonical_classification`). Two drafted questions can ask for the same
    information in different words — for example, a kebab-case identifier and a
    human-readable name for the same data contract. The classifier marks the redundant one
    with `duplicate_of`. This must collapse both into ONE pending question (the one people
    would naturally answer first), and answering it once must fill in the answer for both
    original questions, not just the one actually shown."""
    await _insert_bronze(
        DATE_CLASSIFY_SEMANTIC_DUPLICATE,
        "meeting_d.en.vtt",
        ["We're introducing a new contract between the backend and the website."],
    )
    try:
        name_question = {
            "id": "contract.backend_to_website.name",
            "scope": "data_contract",
            "target": "backend-to-website (frontend)",
            "requirement": "name",
            "question": "What human-readable name should be assigned to the backend-to-website (frontend) contract?",
        }
        id_question = {
            "id": "contract.backend_to_website.id",
            "scope": "data_contract",
            "target": "backend-to-website (frontend)",
            "requirement": "id",
            "question": (
                "What stable kebab-case identifier should be assigned to the "
                "backend-to-website (frontend) contract?"
            ),
        }
        fake = _make_fake_acompletion(
            classification={
                "classifications": [
                    {"id": name_question["id"], "answer": None, "status": "needs_clarification"},
                    {
                        "id": id_question["id"],
                        "answer": None,
                        "status": "needs_clarification",
                        "duplicate_of": name_question["id"],
                    },
                ]
            },
            synthesis_queue=["# Architecture Description / Evolution\n\nContract named checkout-events."],
            critique_queue=[[]],
            generated_questions=[name_question, id_question],
        )
        monkeypatch.setattr(litellm, "acompletion", fake)

        first = await _run("classify-semantic-duplicate-1", initial_state("20260615"))
        assert "__interrupt__" in first
        payload = first["__interrupt__"][0].value
        # Only the name question reaches the human. The id question, marked as its duplicate,
        # never becomes a second pending question.
        assert payload["pending_questions"] == [name_question["question"]]

        from langgraph.types import Command

        second = await _run(
            "classify-semantic-duplicate-1",
            Command(resume={name_question["question"]: "checkout-events"}),
        )
        assert "__interrupt__" not in second

        async with async_session_factory() as session:
            rows = (
                (
                    await session.execute(
                        select(SilverClarification).where(
                            SilverClarification.ingestion_date == DATE_CLASSIFY_SEMANTIC_DUPLICATE
                        )
                    )
                )
                .scalars()
                .all()
            )
        # Both original questions — the surfaced one and its folded-in duplicate — end up
        # answered the same way, even though only one was ever shown to a human.
        assert len(rows) == 2
        assert {row.question for row in rows} == {name_question["question"]}
        assert {row.answer for row in rows} == {"checkout-events"}
    finally:
        await _cleanup_date(DATE_CLASSIFY_SEMANTIC_DUPLICATE)


async def test_contradiction_escalates_once_and_passes_after_human_informed_redraft(monkeypatch):
    await _insert_bronze(DATE_CONTRADICTION, "meeting_c.en.vtt", ["The auth service is being retired."])
    try:
        fake = _make_fake_acompletion(
            classification={"classifications": []},
            synthesis_queue=[
                "# Architecture Description / Evolution\n\nAuth service is deprecated.",
                "# Architecture Description / Evolution\n\nAuth service is active and maintained.",
            ],
            critique_queue=[
                [
                    {
                        "claim": "Auth service is deprecated.",
                        "supported": False,
                        "rationale": "transcript says retired, not deprecated-and-removed",
                        "severity": "material",
                    }
                ],
                [],
            ],
        )
        monkeypatch.setattr(litellm, "acompletion", fake)

        first = await _run("contradiction-1", initial_state("20260603"))
        assert "__interrupt__" in first
        payload = first["__interrupt__"][0].value
        assert payload["origin"] == "boss"
        assert len(payload["pending_questions"]) == 1
        question = payload["pending_questions"][0]
        assert "Auth service is deprecated." in question

        from langgraph.types import Command

        second = await _run(
            "contradiction-1", Command(resume={question: "It's retired, not deprecated."})
        )
        assert "__interrupt__" not in second

        async with async_session_factory() as session:
            doc = (
                (
                    await session.execute(
                        select(SilverDocument).where(SilverDocument.ingestion_date == DATE_CONTRADICTION)
                    )
                )
                .scalars()
                .one()
            )
        assert doc.content == "# Architecture Description / Evolution\n\nAuth service is active and maintained."
    finally:
        await _cleanup_date(DATE_CONTRADICTION)


async def test_duplicate_material_claims_produce_one_pending_question_not_two(monkeypatch):
    """This is a regression test for a real bug. The Critic can flag two distinct claim
    entries with the exact same `claim` text. For example, the same sentence gets quoted
    twice, once per severity read. `boss_decide` used to turn each one into its own
    pending-question string. So two identical question strings could reach `ask_human`. The
    frontend renders one text_input per pending question. At the time, it keyed each one off
    the question text itself. Streamlit crashed outright, with a
    `StreamlitDuplicateElementKey` error, when two widgets shared a key. `_top_questions`, in
    `agents/graph.py`, now removes exact-text duplicates before returning. This test drives
    that same duplicate-claim shape through the real graph. It asserts that exactly one
    question survives."""
    await _insert_bronze(DATE_DUPLICATE_MATERIAL_CLAIMS, "meeting_g.en.vtt", ["The billing service was discussed."])
    try:
        duplicate_claim = "Billing Service now charges customers automatically."
        duplicate_rationale = "transcript never confirms automatic charging"
        fake = _make_fake_acompletion(
            classification={"classifications": []},
            synthesis_queue=[
                "# Architecture Description / Evolution\n\nBilling Service now charges customers automatically."
            ],
            critique_queue=[
                # These two claim entries have byte-identical claim text and rationale. This
                # is what makes boss_decide's own formatted question strings collide exactly.
                # It is the real shape of the bug. A different rationale per entry would
                # already produce two distinct strings, and no de-dupe would be needed at all.
                [
                    {
                        "claim": duplicate_claim,
                        "supported": False,
                        "rationale": duplicate_rationale,
                        "severity": "material",
                    },
                    {
                        "claim": duplicate_claim,
                        "supported": False,
                        "rationale": duplicate_rationale,
                        "severity": "material",
                    },
                ]
            ],
        )
        monkeypatch.setattr(litellm, "acompletion", fake)

        result = await _run("duplicate-material-claims-1", initial_state("20260612"))
        assert "__interrupt__" in result
        pending = result["__interrupt__"][0].value["pending_questions"]

        assert len(pending) == len(set(pending)), f"duplicate pending questions reached ask_human: {pending}"
        assert len(pending) == 1
    finally:
        await _cleanup_date(DATE_DUPLICATE_MATERIAL_CLAIMS)


async def test_low_severity_claim_is_downgraded_without_any_interrupt(monkeypatch):
    await _insert_bronze(DATE_LOW_SEVERITY, "meeting_d.en.vtt", ["The reporting pipeline was discussed."])
    try:
        fake = _make_fake_acompletion(
            classification={"classifications": []},
            synthesis_queue=[
                "# Architecture Description / Evolution\n\nReporting pipeline runs nightly."
            ],
            critique_queue=[
                [
                    {
                        "claim": "runs nightly",
                        "supported": False,
                        "rationale": "frequency not explicitly stated, just plausible",
                        "severity": "low",
                    }
                ]
            ],
        )
        monkeypatch.setattr(litellm, "acompletion", fake)

        result = await _run("low-severity-1", initial_state("20260604"))
        assert "__interrupt__" not in result

        async with async_session_factory() as session:
            doc = (
                (
                    await session.execute(
                        select(SilverDocument).where(SilverDocument.ingestion_date == DATE_LOW_SEVERITY)
                    )
                )
                .scalars()
                .one()
            )
        assert "runs nightly **[unknown — flagged by review]**" in doc.content
    finally:
        await _cleanup_date(DATE_LOW_SEVERITY)


async def test_downgrading_a_claim_never_corrupts_a_mermaid_diagram(monkeypatch):
    """This is a regression test for a real bug. The Critic's claim text is a verbatim quote
    from anywhere in the document. See `prompts/adr_critic/critic.jinja`. This includes
    diagram node labels. When Boss downgraded such a claim, it used to splice
    `**[unknown — flagged by review]**` straight into the ```mermaid fence. That broke the
    diagram's syntax, because Mermaid has no `**` token. This test reproduces exactly that
    case. "integration layer" appears only inside the target-architecture diagram, nowhere in
    prose. `_downgrade_claim`, in `agents/graph.py`, must leave the diagram byte-for-byte
    untouched. It must still downgrade a claim quoted from prose."""
    await _insert_bronze(DATE_MERMAID_DOWNGRADE, "meeting_f.en.vtt", ["The reporting pipeline was discussed."])
    try:
        mermaid_diagram = '    A["integration layer"] --> B["Reporting Service"]'
        fake = _make_fake_acompletion(
            classification={"classifications": []},
            synthesis_queue=[
                "# Architecture Description / Evolution\n\n"
                "The reporting pipeline was migrated to streaming ingestion.\n\n"
                "## Target architecture\n\n"
                "```mermaid\n"
                "graph TD\n"
                f"{mermaid_diagram}\n"
                "```\n"
            ],
            critique_queue=[
                [
                    {
                        "claim": "migrated to streaming ingestion",
                        "supported": False,
                        "rationale": "not explicitly confirmed as a completed migration",
                        "severity": "low",
                    },
                    {
                        "claim": "integration layer",
                        "supported": False,
                        "rationale": "quoted from the diagram node label, not confirmed as a real component",
                        "severity": "low",
                    },
                ]
            ],
        )
        monkeypatch.setattr(litellm, "acompletion", fake)

        result = await _run("mermaid-downgrade-1", initial_state("20260610"))
        assert "__interrupt__" not in result

        async with async_session_factory() as session:
            doc = (
                (
                    await session.execute(
                        select(SilverDocument).where(SilverDocument.ingestion_date == DATE_MERMAID_DOWNGRADE)
                    )
                )
                .scalars()
                .one()
            )

        # The prose claim is downgraded normally...
        assert "migrated to streaming ingestion **[unknown — flagged by review]**" in doc.content
        # But the diagram is untouched. It is the only place "integration layer" appears.
        assert mermaid_diagram in doc.content
        mermaid_block = doc.content.split("```mermaid")[1].split("```")[0]
        assert "**" not in mermaid_block
    finally:
        await _cleanup_date(DATE_MERMAID_DOWNGRADE)


async def test_claim_still_material_after_one_retry_proceeds_without_a_third_ask(monkeypatch):
    await _insert_bronze(DATE_BOUNDED_RETRY, "meeting_e.en.vtt", ["The payments service was mentioned."])
    try:
        fake = _make_fake_acompletion(
            classification={"classifications": []},
            synthesis_queue=[
                "# Architecture Description / Evolution\n\nPayments service handles refunds.",
                "# Architecture Description / Evolution\n\nPayments service handles refunds, still contested.",
            ],
            critique_queue=[
                [
                    {
                        "claim": "handles refunds",
                        "supported": False,
                        "rationale": "transcript never mentions refunds",
                        "severity": "material",
                    }
                ],
                [
                    {
                        "claim": "handles refunds",
                        "supported": False,
                        "rationale": "still not mentioned, even after clarification",
                        "severity": "material",
                    }
                ],
            ],
        )
        monkeypatch.setattr(litellm, "acompletion", fake)

        first = await _run("bounded-retry-1", initial_state("20260605"))
        assert "__interrupt__" in first
        question = first["__interrupt__"][0].value["pending_questions"][0]

        from langgraph.types import Command

        second = await _run("bounded-retry-1", Command(resume={question: "no sé"}))
        # The claim is still flagged after the one allowed retry. The graph proceeds anyway.
        # It never asks again.
        assert "__interrupt__" not in second

        async with async_session_factory() as session:
            doc = (
                (
                    await session.execute(
                        select(SilverDocument).where(SilverDocument.ingestion_date == DATE_BOUNDED_RETRY)
                    )
                )
                .scalars()
                .one()
            )
        assert "handles refunds **[unknown — flagged by review]**" in doc.content
    finally:
        await _cleanup_date(DATE_BOUNDED_RETRY)


async def test_gold_extraction_persists_and_skips_unknown_status(monkeypatch):
    """This is an end-to-end test of `extract_gold_facts`, then `resolve_gold_identity`,
    then `persist_gold_evolution`. See `agents/graph.py` and `.tmp/gold_process_v5.md` §2.
    These run inside this same graph run, right after `chunk_and_embed`. A confirmed
    component gets a `gold_evolution` row and a `gold_aliases` entry. A component the
    extraction leaves as `"unknown"` gets neither. See `agents.graph.persist_gold_evolution`'s
    skip logic. Per v6 §3, `"unknown"` should never reach Gold."""
    source = "meeting_gold.en.vtt"
    await _insert_bronze(DATE_GOLD_EXTRACTION, source, ["The checkout service was introduced today."])
    try:
        gold_extraction = {
            "components": [
                {
                    "name": "Checkout Service",
                    "status": "new",
                    "narrative": "Checkout Service is a new component introduced to handle checkout.",
                    "dependency_names": [],
                    "contract_names": [],
                },
                {
                    "name": "Mystery Service",
                    "status": "unknown",
                    "narrative": "Mentioned but its status could not be determined from the ADR.",
                    "dependency_names": [],
                    "contract_names": [],
                },
            ],
            "contracts": [],
            "architecture_change": "changed",
            "architecture_narrative": "Checkout Service was added to the architecture.",
            "mermaid_diagram": "",
        }
        # There are two entries each. The second `_run` below uses a brand-new thread_id.
        # So the graph replays synthesize_document and critic_document from scratch too, not
        # just the Gold nodes. Both runs use the same text, so write_document's hash-compare
        # keeps `version == 1` on the second run. This is what makes `already_extracted`
        # actually skip Gold's LLM call, rather than just happening to not need it.
        fake = _make_fake_acompletion(
            classification={"classifications": []},
            synthesis_queue=[
                "# ADR — Checkout\n\nCheckout service is new.",
                "# ADR — Checkout\n\nCheckout service is new.",
            ],
            critique_queue=[[], []],
            mentioned_components=[{"name": "checkout service", "status": "new"}],
            gold_extraction=gold_extraction,
        )
        monkeypatch.setattr(litellm, "acompletion", fake)

        result = await _run("gold-extraction-1", initial_state("20260609"))
        assert "__interrupt__" not in result

        async with async_session_factory() as session:
            rows = (
                (
                    await session.execute(
                        select(GoldEvolution).where(GoldEvolution.ingestion_date == DATE_GOLD_EXTRACTION)
                    )
                )
                .scalars()
                .all()
            )
        by_type = {(r.entity_type, r.canonical_name): r for r in rows}

        checkout = by_type[("component", "Checkout Service")]
        assert checkout.version == 1
        assert checkout.operation == "new"
        assert checkout.source_component == source
        assert checkout.source_adr_version == 1

        architecture = by_type[("architecture", source)]
        assert architecture.operation == "changed"

        # "Mystery Service" has status "unknown". It must not appear anywhere in gold_evolution.
        assert ("component", "Mystery Service") not in by_type
        assert len(rows) == 2  # This is exactly checkout plus architecture. Nothing exists for Mystery Service.

        async with async_session_factory() as session:
            aliases = (
                (
                    await session.execute(
                        select(GoldAlias).where(GoldAlias.entity_id == checkout.entity_id)
                    )
                )
                .scalars()
                .all()
            )
        assert [a.alias for a in aliases] == ["Checkout Service"]

        # Re-running the same graph for the same source and version must not re-call the LLM
        # for extraction at all. already_extracted short-circuits it. See agents/graph.py.
        def _explode(*, model, api_key, messages, **kwargs):
            raise AssertionError("extract_gold_facts must not re-call the LLM for an already-processed version")

        async def fake_acompletion_no_gold_call(*, model, api_key, messages, **kwargs):
            content_in = messages[0]["content"]
            if "extracting structured, versionable facts" in content_in:
                _explode(model=model, api_key=api_key, messages=messages, **kwargs)
            return await fake(model=model, api_key=api_key, messages=messages, **kwargs)

        monkeypatch.setattr(litellm, "acompletion", fake_acompletion_no_gold_call)
        await _run("gold-extraction-2", initial_state("20260609"))
    finally:
        await _cleanup_date(DATE_GOLD_EXTRACTION)


async def test_gold_data_contract_resolves_producer_and_consumer_to_component_entity_ids(monkeypatch):
    """Regression/feature test for `DataContractPayload.producer_id`/`consumer_id`. A data
    contract's `producer`/`consumer` are themselves components. Before this, Gold stored only
    their raw names on the contract's own payload — never resolved through `gold_aliases` the
    way `ComponentPayload.dependency_ids`/`contract_ids` already are. This meant there was no
    reliable way to ask "which components does this data contract touch" by a stable id;
    only by matching a name string that a rename could break. `resolve_gold_identity` now
    resolves a contract's producer and consumer as components too (see its own docstring), and
    `_persist_contracts` writes their resolved ids onto the contract's payload."""
    source = "meeting_gold_contract_ids.en.vtt"
    await _insert_bronze(
        DATE_GOLD_CONTRACT_PRODUCER_CONSUMER_IDS,
        source,
        ["Checkout Service now publishes checkout-completed to Loyalty Service."],
    )
    try:
        gold_extraction = {
            "components": [
                {
                    "name": "Checkout Service",
                    "status": "unchanged",
                    "narrative": "Checkout Service handles checkout and now publishes an event.",
                    "dependency_names": [],
                    "contract_names": ["checkout-completed"],
                },
                {
                    "name": "Loyalty Service",
                    "status": "new",
                    "narrative": "Loyalty Service is a new component that awards loyalty points.",
                    "dependency_names": [],
                    "contract_names": [],
                },
            ],
            "contracts": [
                {
                    "name": "checkout-completed",
                    "action": "new",
                    "narrative": "checkout-completed is a new event from Checkout Service to Loyalty Service.",
                    "producer": "Checkout Service",
                    "consumer": "Loyalty Service",
                    "odcs_spec": "{}",
                }
            ],
            "architecture_change": "changed",
            "architecture_narrative": "Loyalty Service was added, connected to Checkout Service.",
            "mermaid_diagram": "",
        }
        fake = _make_fake_acompletion(
            classification={"classifications": []},
            synthesis_queue=["# ADR — Loyalty\n\nLoyalty Service added."],
            critique_queue=[[]],
            mentioned_components=[
                {"name": "checkout service", "status": "unchanged"},
                {"name": "loyalty service", "status": "new"},
            ],
            gold_extraction=gold_extraction,
        )
        monkeypatch.setattr(litellm, "acompletion", fake)

        result = await _run("gold-contract-producer-consumer-ids-1", initial_state("20260617"))
        assert "__interrupt__" not in result

        async with async_session_factory() as session:
            rows = (
                (
                    await session.execute(
                        select(GoldEvolution).where(
                            GoldEvolution.ingestion_date == DATE_GOLD_CONTRACT_PRODUCER_CONSUMER_IDS
                        )
                    )
                )
                .scalars()
                .all()
            )
        by_type = {(r.entity_type, r.canonical_name): r for r in rows}

        checkout = by_type[("component", "Checkout Service")]
        loyalty = by_type[("component", "Loyalty Service")]
        contract = by_type[("data_contract", "checkout-completed")]

        assert contract.payload["producer_id"] == checkout.entity_id
        assert contract.payload["consumer_id"] == loyalty.entity_id
        # Both ids are real, live component entities, not made-up strings.
        assert contract.payload["producer_id"] in {r.entity_id for r in rows if r.entity_type == "component"}
    finally:
        await _cleanup_date(DATE_GOLD_CONTRACT_PRODUCER_CONSUMER_IDS)


async def test_gold_discovers_a_component_never_in_the_pre_clarification_mentioned_list(monkeypatch):
    """This is a regression test for a real bug. A component introduced only through a
    clarification answer, never named in `generate_architecture_questions`'s own
    pre-clarification `mentioned_components`, used to be permanently invisible to Gold. This
    happened because `extract_gold_facts` was grounded against that earlier list. It could
    never extract anything outside it. This was `prompts/gold/extraction.jinja`'s old "fixed
    list, never extract beyond it" rule. Gold now discovers components straight from the
    final, clarified ADR. See `build_gold_extraction_prompt`'s own docstring. This test fakes
    exactly that shape. `mentioned_components` only ever names "Checkout Service". But the
    faked extraction result, standing in for what a real LLM reading the final ADR would
    find, also reports a brand-new "Notification Bus" that the ADR's own clarification
    answers introduced. Both must reach `gold_evolution`. The node must not filter the
    extraction against the earlier list."""
    source = "meeting_new_component.en.vtt"
    await _insert_bronze(
        DATE_GOLD_DISCOVERS_NEW_COMPONENT, source, ["Checkout Service handles order checkout flows."]
    )
    try:
        gold_extraction = {
            "components": [
                {
                    "name": "Checkout Service",
                    "status": "unchanged",
                    "narrative": "Checkout Service continues to handle order checkout flows.",
                    "dependency_names": [],
                    "contract_names": [],
                },
                {
                    "name": "Notification Bus",
                    "status": "new",
                    "narrative": "A new asynchronous message broker introduced via clarification, "
                    "never named in the original transcript.",
                    "dependency_names": [],
                    "contract_names": [],
                },
            ],
            "contracts": [],
            "architecture_change": "changed",
            "architecture_narrative": "A Notification Bus was added to the architecture.",
            "mermaid_diagram": "",
        }
        fake = _make_fake_acompletion(
            classification={"classifications": []},
            synthesis_queue=["# ADR — Checkout\n\nA Notification Bus was introduced via clarification."],
            critique_queue=[[]],
            # Only "Checkout Service" was ever identified before clarification happened.
            # "Notification Bus" is not in this list, and must never need to be.
            mentioned_components=[{"name": "checkout service", "status": "unchanged"}],
            gold_extraction=gold_extraction,
        )
        monkeypatch.setattr(litellm, "acompletion", fake)

        result = await _run("gold-discovers-new-component-1", initial_state("20260611"))
        assert "__interrupt__" not in result

        async with async_session_factory() as session:
            rows = (
                (
                    await session.execute(
                        select(GoldEvolution).where(
                            GoldEvolution.ingestion_date == DATE_GOLD_DISCOVERS_NEW_COMPONENT
                        )
                    )
                )
                .scalars()
                .all()
            )
        by_name = {r.canonical_name: r for r in rows if r.entity_type == "component"}

        assert "Notification Bus" in by_name, (
            "a component only named in the (faked) final-ADR extraction, never in the "
            "pre-clarification mentioned_components list, must still reach gold_evolution"
        )
        assert by_name["Notification Bus"].operation == "new"
        assert by_name["Checkout Service"].operation == "unchanged"
    finally:
        await _cleanup_date(DATE_GOLD_DISCOVERS_NEW_COMPONENT)


async def test_irrelevant_marker_from_the_ui_is_treated_as_a_decline(monkeypatch):
    """The frontend's "Irrelevant" question-action button, in
    `frontend/app.py::_render_question_row`, sends the literal string "[IRRELEVANT]" as the
    answer, instead of typed text. `ask_human` must fold it into `_DECLINE_PHRASES`,
    case-insensitively, the same way it handles "unknown" and "n/a". So it ends up recorded
    as no answer at all, not as literal answer text."""
    await _insert_bronze(DATE_IRRELEVANT_MARKER, "meeting_irrelevant.en.vtt", ["Someone mentioned a detail."])
    try:
        fake = _make_fake_acompletion(
            classification={
                "classifications": [
                    {"id": "component.detail.owner", "answer": None, "status": "needs_clarification"}
                ]
            },
            synthesis_queue=["# Architecture Description / Evolution\n\nDetail: unresolved."],
            critique_queue=[[]],
            generated_questions=[
                {
                    "id": "component.detail.owner",
                    "scope": "component",
                    "target": "detail",
                    "requirement": "owner",
                    "question": "does this detail matter?",
                }
            ],
        )
        monkeypatch.setattr(litellm, "acompletion", fake)

        first = await _run("irrelevant-marker-1", initial_state("20260614"))
        assert "__interrupt__" in first

        from langgraph.types import Command

        second = await _run("irrelevant-marker-1", Command(resume={"does this detail matter?": "[IRRELEVANT]"}))
        assert "__interrupt__" not in second

        async with async_session_factory() as session:
            clarification_row = (
                (
                    await session.execute(
                        select(SilverClarification).where(
                            SilverClarification.ingestion_date == DATE_IRRELEVANT_MARKER
                        )
                    )
                )
                .scalars()
                .one()
            )
        assert clarification_row.answer is None, (
            "the [IRRELEVANT] marker must be folded into a decline, never persisted as literal "
            "answer text"
        )
    finally:
        await _cleanup_date(DATE_IRRELEVANT_MARKER)


def test_ensure_schema_questions_injects_a_fallback_for_a_contract_with_none_drafted():
    contracts = [{"name": "OrderCreated", "producer": "OrderService", "consumer": "FulfillmentService", "action": "new"}]
    questions = [
        {
            "id": "contract.ordercreated.version",
            "scope": "data_contract",
            "target": "OrderCreated",
            "requirement": "version",
            "question": "What version is OrderCreated?",
        }
    ]

    result = _ensure_schema_questions(questions, contracts)

    schema_questions = [q for q in result if "schema" in q["id"]]
    assert len(schema_questions) == 1
    assert schema_questions[0]["target"] == "OrderCreated"
    assert schema_questions[0]["scope"] == "data_contract"
    assert len(result) == len(questions) + 1


def test_ensure_schema_questions_does_not_duplicate_an_existing_schema_question():
    contracts = [{"name": "OrderCreated", "producer": "OrderService", "consumer": "FulfillmentService", "action": "new"}]
    questions = [
        {
            "id": "contract.ordercreated.schema.order_id.type",
            "scope": "data_contract",
            "target": "OrderCreated",
            "requirement": "schema.order_id.type",
            "question": "What is the data type of order_id in OrderCreated?",
        }
    ]

    result = _ensure_schema_questions(questions, contracts)

    assert result == questions  # Nothing is added, because a schema question already exists.


def test_drop_new_contract_version_questions_removes_version_question_for_new_contract():
    contracts = [
        {"name": "OrderCreated", "producer": "OrderService", "consumer": "FulfillmentService", "action": "new"}
    ]
    questions = [
        {
            "id": "contract.ordercreated.version",
            "scope": "data_contract",
            "target": "OrderCreated",
            "requirement": "version",
            "question": "What version is OrderCreated?",
        },
        {
            "id": "contract.ordercreated.schema",
            "scope": "data_contract",
            "target": "OrderCreated",
            "requirement": "schema",
            "question": "What fields does OrderCreated carry?",
        },
    ]

    result = _drop_new_contract_version_questions(questions, contracts)

    assert [q["id"] for q in result] == ["contract.ordercreated.schema"]


def test_drop_new_contract_version_questions_keeps_version_question_for_forward_update_contract():
    """Versioning IS a real, askable question for a contract that already existed before this
    change — only a brand-new contract's version is a fixed convention."""
    contracts = [
        {
            "name": "ProcessedEvent",
            "producer": "EventProcessor",
            "consumer": "AnalyticsService",
            "action": "forward-update",
        }
    ]
    questions = [
        {
            "id": "contract.processedevent.version",
            "scope": "data_contract",
            "target": "ProcessedEvent",
            "requirement": "version",
            "question": "What was the previous version and what is the new version of ProcessedEvent?",
        }
    ]

    result = _drop_new_contract_version_questions(questions, contracts)

    assert result == questions


async def test_synthesize_document_retries_with_a_named_correction_when_suggest_info_is_dropped(monkeypatch):
    """Regression test for a real, reported bug: a reviewer answers a clarification
    `[SUGGEST INFO]` (e.g. "how do the frontend and backend interact?"), but the model
    sometimes drops it silently — no `LLM SUGGESTION:` anywhere, no diagram edge, leaving two
    brand-new components disconnected in the ADR's own diagram (and, downstream, in Gold's live
    architecture diagram too). `agents.graph.synthesize_document`'s retry loop must, on this
    specific failure, send a follow-up turn naming the exact gap
    (`SUGGEST_INFO_CORRECTION_MESSAGE`) rather than only blindly resampling — real testing
    showed a blind resample sometimes needs 3-4 attempts to recover, while a named correction
    converged in one."""
    dropped_document = "# ADR — Marketplace\n\nFrontend and backend are both introduced."
    corrected_document = (
        "# ADR — Marketplace\n\nFrontend and backend are both introduced.\n\n"
        "LLM SUGGESTION: The frontend calls the backend over a REST API."
    )
    calls: list[list[dict]] = []

    async def fake_acompletion(*, model, api_key, messages, **kwargs):
        calls.append(messages)
        content = dropped_document if len(calls) == 1 else corrected_document
        return _fake_response(content)

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    state = initial_state("20260919")
    state["bronze_documents"] = [{"source_component": "market.en.vtt", "content": "backend and frontend"}]
    state["mentioned_components"] = [
        {"name": "frontend", "status": "new"},
        {"name": "backend", "status": "new"},
    ]
    state["clarifications"] = [
        {
            "id": "architecture.interface",
            "scope": "architecture",
            "target": "frontend-backend interaction",
            "requirement": "technical interface",
            "question": "How do the frontend and backend interact?",
            "answer": "[SUGGEST INFO]",
            "status": "answered",
        }
    ]

    result = await synthesize_document(state)

    assert result["documents"]["market.en.vtt"] == corrected_document
    assert len(calls) == 2
    assert calls[1][-1]["content"] == SUGGEST_INFO_CORRECTION_MESSAGE
    assert calls[1][0]["content"] == calls[0][0]["content"]
