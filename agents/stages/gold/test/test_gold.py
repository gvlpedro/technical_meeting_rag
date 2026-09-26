"""Gold layer storage tests, for `agents/stages/gold/service.py`, run against real Postgres.
This file mirrors the style of `tests/test_silver_storage.py`. No LLM is involved anywhere in
this file. Identity resolution, hashing, and versioning are all deterministic, DB-only logic.
See `gold_process.md` §3 and §5. `tests/test_clarification_loop.py`'s Gold-extension test
covers the graph-level, faked-LLM path end to end.
"""

import asyncio
from datetime import date
from types import SimpleNamespace
from uuid import uuid4

import pytest
from sqlalchemy import delete, select

from agents.stages.gold.schemas import (
    ArchitecturePayload,
    ComponentPayload,
    DataContractPayload,
    ExtractedComponent,
    ExtractedDataContract,
    GoldExtractionResult,
)
from agents.stages.gold.service import (
    _dedupe_ids_by_entity,
    _full_odcs_spec_block,
    _payload_detail,
    _reciprocal_rank_fusion,
    _rerank_ids,
    already_extracted,
    contracts_with_real_changes,
    current_architecture_diagram,
    current_gold_state,
    embed_question,
    ensure_alias,
    entity_history,
    extract_and_persist_gold_facts,
    find_entity_by_name_in_text,
    is_evolution_question,
    parse_odcs_spec,
    persist_entity_version,
    resolve_entity_id,
    resolve_entity_id_for_lookup,
    top_k_gold_evolution,
    wants_full_spec,
)
from db.models import GoldAlias, GoldEvolution
from db.session import async_session_factory
from ingestion.reranker import score_candidates

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _unique_component_name() -> str:
    # We use real names, not uuids, so word-boundary-shaped inputs stay realistic. We just
    # make each name collision-free per test run. gold_aliases has no per-test isolation
    # column, so we cannot scope a WHERE clause the way bronze/silver tests scope by
    # ingestion_date.
    return f"Checkout Service {uuid4().hex[:8]}"


async def _cleanup_entity(entity_type: str, entity_id: str) -> None:
    async with async_session_factory() as session:
        await session.execute(
            delete(GoldEvolution).where(GoldEvolution.entity_type == entity_type, GoldEvolution.entity_id == entity_id)
        )
        await session.execute(
            delete(GoldAlias).where(GoldAlias.entity_type == entity_type, GoldAlias.entity_id == entity_id)
        )
        await session.commit()


def test_component_payload_round_trips_dependency_and_contract_ids():
    payload = ComponentPayload(dependency_ids=["e1", "e2"], contract_ids=["c1"])
    assert payload.model_dump() == {
        "dependency_ids": ["e1", "e2"],
        "contract_ids": ["c1"],
        "input_contract_ids": [],
        "output_contract_ids": [],
    }


def test_component_payload_defaults_to_empty_lists():
    assert ComponentPayload().model_dump() == {
        "dependency_ids": [],
        "contract_ids": [],
        "input_contract_ids": [],
        "output_contract_ids": [],
    }


def test_data_contract_payload_keeps_odcs_spec_opaque():
    spec = {"apiVersion": "v3.1.0", "schema": {"type": "object", "properties": {"a": {"type": "string"}}}}
    payload = DataContractPayload(producer="checkout", consumer="billing", odcs_spec=spec)
    assert payload.model_dump()["odcs_spec"] == spec  # It is stored as given, not destructured or validated.


def test_data_contract_payload_defaults_producer_and_consumer_ids_to_empty_string():
    """A contract whose producer/consumer were never resolved — for example, a payload
    persisted before `producer_id`/`consumer_id` existed — must still validate, with an
    honest "not resolved" empty string, not a crash or a made-up id."""
    payload = DataContractPayload(producer="checkout", consumer="billing")
    assert payload.producer_id == ""
    assert payload.consumer_id == ""


def test_data_contract_payload_round_trips_producer_and_consumer_ids():
    producer_id, consumer_id = str(uuid4()), str(uuid4())
    payload = DataContractPayload(
        producer="checkout", consumer="billing", producer_id=producer_id, consumer_id=consumer_id
    )
    assert payload.model_dump()["producer_id"] == producer_id
    assert payload.model_dump()["consumer_id"] == consumer_id


def test_extracted_data_contract_takes_odcs_spec_as_a_string():
    """`ExtractedDataContract` is the `response_format` for the Gold-extraction LLM call.
    It must keep `odcs_spec` as `str`, never `dict`. OpenAI's strict structured-output mode
    rejects any object-typed field that has no `additionalProperties: false`. A deliberately
    open ODCS blob can never declare that. If `odcs_spec` regresses to `dict`, every real
    Gold extraction call that reaches OpenAI breaks. No error appears until the Anthropic
    fallback also fails."""
    contract = ExtractedDataContract(
        name="checkout-events",
        action="new",
        narrative="...",
        producer="checkout",
        consumer="billing",
        odcs_spec='{"apiVersion": "v3.1.0"}',
    )
    assert contract.odcs_spec == '{"apiVersion": "v3.1.0"}'
    with pytest.raises(Exception):
        ExtractedDataContract(
            name="x", action="new", narrative="...", producer="a", consumer="b",
            odcs_spec={"apiVersion": "v3.1.0"},  # A real dict must be rejected here, not silently accepted.
        )


def test_parse_odcs_spec_round_trips_valid_json():
    assert parse_odcs_spec('{"apiVersion": "v3.1.0"}') == {"apiVersion": "v3.1.0"}


def test_parse_odcs_spec_falls_back_to_empty_dict_on_garbage():
    assert parse_odcs_spec("") == {}
    assert parse_odcs_spec("not json") == {}
    assert parse_odcs_spec("[1, 2, 3]") == {}  # This is valid JSON, but not an object. So it still returns {}.


def test_architecture_payload_defaults():
    assert ArchitecturePayload().model_dump() == {
        "mermaid_diagram": "",
        "components": [],
        "dependencies": [],
    }


def test_payload_detail_component_shows_input_and_output_contracts_split():
    row = SimpleNamespace(
        entity_type="component",
        entity_id="checkout-service",
        payload={
            "dependency_ids": [], "contract_ids": ["c1", "c2"],
            "input_contract_ids": ["c1"], "output_contract_ids": ["c2"],
        },
    )
    names = {("data_contract", "c1"): "payment-events", ("data_contract", "c2"): "checkout-events"}
    assert _payload_detail(row, names) == " (input contracts: payment-events; output contracts: checkout-events)"


def test_payload_detail_component_falls_back_to_undifferentiated_contracts_for_pre_migration_rows():
    """A component persisted before the input/output split existed has `contract_ids` but no
    (or empty) `input_contract_ids`/`output_contract_ids` — must still show something, not go
    silent. See `.tmp/improve_timeline_questions_and_linage.md` §2.2, §6."""
    row = SimpleNamespace(
        entity_type="component",
        entity_id="checkout-service",
        payload={"dependency_ids": [], "contract_ids": ["c1"]},  # no split keys at all, like an old row
    )
    names = {("data_contract", "c1"): "checkout-events"}
    assert _payload_detail(row, names) == " (data contracts: checkout-events)"


def test_payload_detail_component_shows_successors():
    row = SimpleNamespace(
        entity_type="component", entity_id="payments-gateway",
        payload={"dependency_ids": [], "contract_ids": []},
    )
    successors = {"payments-gateway": ["Checkout Service", "Refund Service"]}
    assert _payload_detail(row, {}, successors) == " (depended on by: Checkout Service, Refund Service)"


def test_wants_full_spec_recognizes_english_and_spanish_markers():
    assert wants_full_spec("Could you show me the specification for this contract?")
    assert wants_full_spec("What's the schema of registration-login-purchase?")
    assert wants_full_spec("¿Cuál es la especificación completa del contrato?")
    assert not wants_full_spec("What does the Payments Gateway do?")


def test_full_odcs_spec_block_renders_the_entire_spec_for_a_data_contract_row():
    spec = {"apiVersion": "odcs/v3.0.0", "status": "active", "schema": {"properties": {"a": {"type": "string"}}}}
    row = SimpleNamespace(entity_type="data_contract", payload={"producer": "x", "consumer": "y", "odcs_spec": spec})
    block = _full_odcs_spec_block(row)
    assert "Full specification:" in block
    assert '"apiVersion": "odcs/v3.0.0"' in block


def test_full_odcs_spec_block_is_empty_for_a_component_or_an_unpopulated_spec():
    component_row = SimpleNamespace(entity_type="component", payload={"dependency_ids": [], "contract_ids": []})
    assert _full_odcs_spec_block(component_row) == ""

    empty_spec_row = SimpleNamespace(entity_type="data_contract", payload={"producer": "x", "consumer": "y"})
    assert _full_odcs_spec_block(empty_spec_row) == ""


async def test_resolve_entity_id_exact_match_returns_existing_entity_id():
    entity_type = "component"
    entity_id = str(uuid4())
    name = _unique_component_name()
    try:
        async with async_session_factory() as session:
            await ensure_alias(session, entity_type, entity_id, name, "meeting.en.vtt", 1)
            await session.commit()

        async with async_session_factory() as session:
            resolved = await resolve_entity_id(session, entity_type, name)
        assert resolved == entity_id
    finally:
        await _cleanup_entity(entity_type, entity_id)


async def test_resolve_entity_id_fuzzy_match_above_threshold():
    entity_type = "component"
    entity_id = str(uuid4())
    canonical = f"Checkout Service {uuid4().hex[:8]}"
    misspelled = canonical.replace("Checkout", "Checkot")  # This is one typo. It still has more than 0.6 trigram similarity.
    try:
        async with async_session_factory() as session:
            await ensure_alias(session, entity_type, entity_id, canonical, "meeting.en.vtt", 1)
            await session.commit()

        async with async_session_factory() as session:
            resolved = await resolve_entity_id(session, entity_type, misspelled)
        assert resolved == entity_id
    finally:
        await _cleanup_entity(entity_type, entity_id)


async def test_resolve_entity_id_mints_new_id_when_nothing_matches():
    entity_type = "component"
    name = _unique_component_name()
    async with async_session_factory() as session:
        resolved = await resolve_entity_id(session, entity_type, name)
    assert resolved  # This is a fresh ULID string, not empty.
    # This is never inserted anywhere. resolve_entity_id only resolves. ensure_alias inserts.
    async with async_session_factory() as session:
        rows = await session.execute(
            select(GoldAlias).where(GoldAlias.entity_id == resolved)
        )
        assert rows.first() is None


async def test_resolve_entity_id_mints_a_ulid_not_a_uuid4():
    """A freshly minted `entity_id` must be a ULID (`python-ulid`, pinned to 4.0.1): 26
    characters, Crockford-base32 (digits and uppercase letters only, no `I`/`L`/`O`/`U`, and
    no hyphens) — never a `uuid4`'s 36-character hyphenated hex form. This is what lets the
    id double as a short, readable token in a component's own metadata and in a data
    contract's `producer_id`/`consumer_id`, per `resolve_entity_id`'s own docstring."""
    name = _unique_component_name()
    async with async_session_factory() as session:
        resolved = await resolve_entity_id(session, "component", name)
    assert len(resolved) == 26
    assert "-" not in resolved
    assert set(resolved) <= set("0123456789ABCDEFGHJKMNPQRSTVWXYZ")


async def test_ensure_alias_is_idempotent():
    entity_type = "component"
    entity_id = str(uuid4())
    name = _unique_component_name()
    try:
        async with async_session_factory() as session:
            await ensure_alias(session, entity_type, entity_id, name, "meeting.en.vtt", 1)
            await ensure_alias(session, entity_type, entity_id, name, "meeting.en.vtt", 1)  # This is the same triple, called twice.
            await session.commit()

        async with async_session_factory() as session:
            rows = (
                await session.execute(
                    select(GoldAlias).where(
                        GoldAlias.entity_type == entity_type, GoldAlias.entity_id == entity_id
                    )
                )
            ).all()
        assert len(rows) == 1
    finally:
        await _cleanup_entity(entity_type, entity_id)


async def test_persist_entity_version_first_write_is_version_one():
    entity_type = "component"
    entity_id = str(uuid4())
    try:
        async with async_session_factory() as session:
            version = await persist_entity_version(
                session,
                entity_type=entity_type,
                entity_id=entity_id,
                canonical_name="Checkout Service",
                operation="new",
                narrative="Checkout Service is a new component that handles checkout flows.",
                payload={"dependency_ids": [], "contract_ids": []},
                source_component="meeting.en.vtt",
                source_adr_version=1,
                ingestion_date=date(2026, 6, 1),
            )
            await session.commit()
        assert version == 1
    finally:
        await _cleanup_entity(entity_type, entity_id)


async def test_persist_entity_version_same_hash_is_a_noop():
    entity_type = "component"
    entity_id = str(uuid4())
    kwargs = dict(
        entity_type=entity_type,
        entity_id=entity_id,
        canonical_name="Checkout Service",
        operation="unchanged",
        narrative="Checkout Service still handles checkout flows the same way.",
        payload={"dependency_ids": [], "contract_ids": []},
        source_component="meeting.en.vtt",
        source_adr_version=1,
        ingestion_date=date(2026, 6, 1),
    )
    try:
        async with async_session_factory() as session:
            first = await persist_entity_version(session, **kwargs)
            await session.commit()
        assert first == 1

        async with async_session_factory() as session:
            second = await persist_entity_version(session, **{**kwargs, "source_adr_version": 2})
            await session.commit()
        assert second is None  # operation, narrative, and payload are identical. So no new row is created.

        async with async_session_factory() as session:
            rows = (
                await session.execute(
                    select(GoldEvolution).where(
                        GoldEvolution.entity_type == entity_type, GoldEvolution.entity_id == entity_id
                    )
                )
            ).all()
        assert len(rows) == 1
    finally:
        await _cleanup_entity(entity_type, entity_id)


async def test_persist_entity_version_different_hash_bumps_version():
    entity_type = "component"
    entity_id = str(uuid4())
    base_kwargs = dict(
        entity_type=entity_type,
        entity_id=entity_id,
        canonical_name="Checkout Service",
        payload={"dependency_ids": [], "contract_ids": []},
        source_component="meeting.en.vtt",
        source_adr_version=1,
        ingestion_date=date(2026, 6, 1),
    )
    try:
        async with async_session_factory() as session:
            first = await persist_entity_version(
                session, operation="new", narrative="Checkout Service is new.", **base_kwargs
            )
            await session.commit()
        assert first == 1

        async with async_session_factory() as session:
            second = await persist_entity_version(
                session,
                operation="modified",
                narrative="Checkout Service now also validates payment methods.",
                **{**base_kwargs, "source_adr_version": 2},
            )
            await session.commit()
        assert second == 2

        async with async_session_factory() as session:
            rows = (
                await session.execute(
                    select(GoldEvolution)
                    .where(GoldEvolution.entity_type == entity_type, GoldEvolution.entity_id == entity_id)
                    .order_by(GoldEvolution.version)
                )
            ).scalars().all()
        assert len(rows) == 2  # Both versions are kept, narrative/payload untouched.
        assert rows[0].valid_to == date(2026, 6, 1)  # Closed the instant v2 (v1's own superseder) was written.
        assert rows[1].valid_to is None  # v2 is still current.
    finally:
        await _cleanup_entity(entity_type, entity_id)


async def test_contracts_with_real_changes_excludes_unchanged_and_unknown():
    name_to_id = {"a": "id-a", "b": "id-b", "c": "id-c"}
    contracts = [
        {"name": "a", "action": "new"},
        {"name": "b", "action": "unchanged"},
        {"name": "c", "action": "unknown"},
    ]
    assert contracts_with_real_changes(contracts, name_to_id) == {"id-a"}


async def test_persist_entity_version_unchanged_with_same_payload_is_a_noop_despite_reworded_narrative():
    """Regression: `doc/cicle_evolution.md` — a real bug where a component/contract picked up a
    spurious new version every ADR purely because the LLM re-narrates "still unchanged" with
    different wording each time, even though nothing about it actually changed. The FIRST
    "new" -> "unchanged" transition still versions (that first confirmation is itself a real,
    one-time fact worth recording, and `_entity_hash` deliberately includes `operation`) — the
    bug, and this fix, is specifically about a SECOND, THIRD, ... consecutive "unchanged" on top
    of an already-"unchanged" version, exactly the `product-catalog` v2->v3->v4 pattern found in
    real data."""
    entity_type = "component"
    entity_id = str(uuid4())
    payload = {"dependency_ids": [], "contract_ids": [], "input_contract_ids": [], "output_contract_ids": []}
    try:
        async with async_session_factory() as session:
            first = await persist_entity_version(
                session, entity_type=entity_type, entity_id=entity_id, canonical_name="Checkout Service",
                operation="new", narrative="Checkout Service is new.", payload=payload,
                source_component="meeting.en.vtt", source_adr_version=1, ingestion_date=date(2026, 6, 1),
            )
            await session.commit()
        assert first == 1

        async with async_session_factory() as session:
            second = await persist_entity_version(
                session, entity_type=entity_type, entity_id=entity_id, canonical_name="Checkout Service",
                operation="unchanged", narrative="Checkout Service remains the same.", payload=payload,
                source_component="meeting.en.vtt", source_adr_version=2, ingestion_date=date(2026, 6, 2),
            )
            await session.commit()
        assert second == 2  # "new" -> "unchanged" is a real, one-time confirmation — still versions.

        async with async_session_factory() as session:
            third = await persist_entity_version(
                session, entity_type=entity_type, entity_id=entity_id, canonical_name="Checkout Service",
                operation="unchanged", narrative="Checkout Service remains exactly the same as before.",
                payload=payload, source_component="meeting.en.vtt", source_adr_version=3,
                ingestion_date=date(2026, 6, 3),
            )
            await session.commit()
        assert third is None  # "unchanged" repeated, different wording, same payload — no v3.

        async with async_session_factory() as session:
            rows = (
                await session.execute(
                    select(GoldEvolution).where(
                        GoldEvolution.entity_type == entity_type, GoldEvolution.entity_id == entity_id
                    )
                )
            ).scalars().all()
        assert len(rows) == 2
    finally:
        await _cleanup_entity(entity_type, entity_id)


async def test_persist_entity_version_corrects_unchanged_to_modified_when_a_components_own_payload_changed():
    entity_type = "component"
    entity_id = str(uuid4())
    try:
        async with async_session_factory() as session:
            await persist_entity_version(
                session, entity_type=entity_type, entity_id=entity_id, canonical_name="Checkout Service",
                operation="new", narrative="Checkout Service is new.",
                payload={"dependency_ids": [], "contract_ids": [], "input_contract_ids": [], "output_contract_ids": []},
                source_component="meeting.en.vtt", source_adr_version=1, ingestion_date=date(2026, 6, 1),
            )
            await session.commit()

        async with async_session_factory() as session:
            second = await persist_entity_version(
                session, entity_type=entity_type, entity_id=entity_id, canonical_name="Checkout Service",
                operation="unchanged",  # the LLM's own judgment — wrong, per the payload below
                narrative="Checkout Service remains the same.",
                payload={
                    "dependency_ids": [], "contract_ids": ["c1"], "input_contract_ids": ["c1"],
                    "output_contract_ids": [],
                },
                source_component="meeting.en.vtt", source_adr_version=2, ingestion_date=date(2026, 6, 2),
            )
            await session.commit()
        assert second == 2

        async with async_session_factory() as session:
            row = (
                await session.execute(
                    select(GoldEvolution).where(
                        GoldEvolution.entity_type == entity_type, GoldEvolution.entity_id == entity_id,
                        GoldEvolution.version == 2,
                    )
                )
            ).scalars().one()
        assert row.operation == "modified"  # corrected — the payload changed, so "unchanged" was wrong
    finally:
        await _cleanup_entity(entity_type, entity_id)


async def test_persist_entity_version_never_corrects_a_data_contracts_own_operation():
    """`doc/cicle_evolution.md` is explicit that this correction is components-only: a data
    contract's `unchanged` -> real-change judgment (`forward-update` vs `break-change`) needs
    semantic understanding of the schema diff a payload comparison cannot provide. A changed
    payload here must still bump the version (this is not a regression of the noise fix — a
    genuine payload difference is a genuine reason to version), it just must not silently invent
    a specific action the LLM never asserted."""
    entity_type = "data_contract"
    entity_id = str(uuid4())
    try:
        async with async_session_factory() as session:
            await persist_entity_version(
                session, entity_type=entity_type, entity_id=entity_id, canonical_name="checkout-events",
                operation="new", narrative="checkout-events is new.",
                payload={"producer": "a", "consumer": "b", "odcs_spec": {}},
                source_component="meeting.en.vtt", source_adr_version=1, ingestion_date=date(2026, 6, 1),
            )
            await session.commit()

        async with async_session_factory() as session:
            second = await persist_entity_version(
                session, entity_type=entity_type, entity_id=entity_id, canonical_name="checkout-events",
                operation="unchanged", narrative="checkout-events remains the same.",
                payload={"producer": "a", "consumer": "b", "odcs_spec": {"apiVersion": "v2"}},
                source_component="meeting.en.vtt", source_adr_version=2, ingestion_date=date(2026, 6, 2),
            )
            await session.commit()
        assert second == 2  # payload changed, so it still versions...

        async with async_session_factory() as session:
            row = (
                await session.execute(
                    select(GoldEvolution).where(
                        GoldEvolution.entity_type == entity_type, GoldEvolution.entity_id == entity_id,
                        GoldEvolution.version == 2,
                    )
                )
            ).scalars().one()
        assert row.operation == "unchanged"  # ...but `operation` is left exactly as the LLM asserted it
    finally:
        await _cleanup_entity(entity_type, entity_id)


async def test_already_extracted_true_after_a_write_false_before():
    entity_type = "component"
    entity_id = str(uuid4())
    source = f"meeting-{uuid4().hex[:8]}.en.vtt"
    try:
        async with async_session_factory() as session:
            before = await already_extracted(session, source, 1)
        assert before is False

        async with async_session_factory() as session:
            await persist_entity_version(
                session,
                entity_type=entity_type,
                entity_id=entity_id,
                canonical_name="Checkout Service",
                operation="new",
                narrative="Checkout Service is new.",
                payload={"dependency_ids": [], "contract_ids": []},
                source_component=source,
                source_adr_version=1,
                ingestion_date=date(2026, 6, 1),
            )
            await session.commit()

        async with async_session_factory() as session:
            after = await already_extracted(session, source, 1)
        assert after is True
    finally:
        await _cleanup_entity(entity_type, entity_id)


async def test_extract_and_persist_gold_facts_serializes_concurrent_calls_for_the_same_adr_version(
    monkeypatch,
):
    """Regression test for a real production bug: the frontend's "Publish" button used to run
    `finalize_document` inline, with no in-progress guard — unlike every other action button in
    `frontend/app.py`. A genuine double-click could fire two overlapping `POST /transcriptions/
    finalize` requests. Both used to read `already_extracted() == False` before either had
    committed (a plain check-then-act, no row lock), so both ran their own LLM extraction and
    both called `persist_entity_version` for every entity — which only no-ops on an EXACT hash
    match, so two independently phrased extractions produced a spurious version 2 for entities
    that should have had exactly one version. This was observed in production as a component,
    both its data contracts, and the architecture entity all showing a v1 AND a v2 from what was
    a single Publish click.

    This fakes `extract_gold_facts_for_source` to return a DIFFERENT narrative on every call —
    deliberately the worst case, guaranteeing a version bump if the two calls are not
    serialized — then fires two concurrent `extract_and_persist_gold_facts` calls for the exact
    same `(tenant, source_component, source_adr_version)`, each on its own `AsyncSession`, the
    same way two overlapping HTTP requests each get their own session. `extract_and_persist_
    gold_facts`'s `pg_advisory_xact_lock` must serialize them: exactly one call should actually
    extract, and exactly one `gold_evolution` version should exist afterward."""
    tenant = f"test-race-{uuid4().hex[:8]}"
    source_component = f"race-{uuid4().hex[:8]}.en.vtt"
    call_count = 0

    async def fake_extract(adr_content: str) -> GoldExtractionResult:
        nonlocal call_count
        call_count += 1
        return GoldExtractionResult(
            components=[
                ExtractedComponent(
                    name="Race Service", status="new", narrative=f"narrative variant {call_count}"
                )
            ],
            contracts=[],
            architecture_change="changed",
            architecture_narrative=f"architecture narrative variant {call_count}",
        )

    monkeypatch.setattr("agents.stages.gold.service.extract_gold_facts_for_source", fake_extract)

    async def _run() -> bool:
        async with async_session_factory() as session:
            extracted = await extract_and_persist_gold_facts(
                session, "adr content", source_component, 1, date(2026, 6, 1), tenant=tenant
            )
            await session.commit()
            return extracted

    results = await asyncio.gather(_run(), _run())
    assert sorted(results) == [False, True]  # exactly one of the two calls actually extracted

    async with async_session_factory() as session:
        rows = (
            await session.execute(
                select(GoldEvolution).where(
                    GoldEvolution.tenant == tenant,
                    GoldEvolution.source_component == source_component,
                    GoldEvolution.canonical_name == "Race Service",
                )
            )
        ).scalars().all()
    assert len(rows) == 1
    assert rows[0].version == 1


async def test_extract_and_persist_corrects_unchanged_to_modified_when_its_own_contract_is_new(monkeypatch):
    """End-to-end regression, through the real call site (not a direct `persist_entity_version`
    call): `doc/cicle_evolution.md` "Regla especial", condition 2 — a component the LLM marks
    "unchanged" is corrected to "modified" when a data contract that names it as producer or
    consumer is itself extracted as genuinely new/changed in this SAME round, even though the
    component's own narrative and its own list of contract ids says nothing new."""
    tenant = f"test-evolution-{uuid4().hex[:8]}"
    source_component = f"evolution-{uuid4().hex[:8]}.en.vtt"

    async def fake_extract_v1(adr_content: str) -> GoldExtractionResult:
        return GoldExtractionResult(
            components=[
                ExtractedComponent(name="Checkout Service", status="new", narrative="Checkout Service is new.")
            ],
            contracts=[],
            architecture_change="changed",
            architecture_narrative="Checkout Service was introduced.",
        )

    monkeypatch.setattr("agents.stages.gold.service.extract_gold_facts_for_source", fake_extract_v1)
    try:
        async with async_session_factory() as session:
            await extract_and_persist_gold_facts(
                session, "adr content v1", source_component, 1, date(2026, 6, 1), tenant=tenant
            )
            await session.commit()

        async def fake_extract_v2(adr_content: str) -> GoldExtractionResult:
            return GoldExtractionResult(
                components=[
                    ExtractedComponent(
                        name="Checkout Service", status="unchanged",
                        narrative="Checkout Service remains the same.",
                    )
                ],
                contracts=[
                    ExtractedDataContract(
                        name="checkout-completed", action="new",
                        narrative="checkout-completed is a new event from Checkout Service.",
                        producer="Checkout Service", consumer="Loyalty Service",
                    )
                ],
                architecture_change="changed",
                architecture_narrative="checkout-completed was added.",
            )

        monkeypatch.setattr("agents.stages.gold.service.extract_gold_facts_for_source", fake_extract_v2)
        async with async_session_factory() as session:
            await extract_and_persist_gold_facts(
                session, "adr content v2", source_component, 2, date(2026, 6, 2), tenant=tenant
            )
            await session.commit()

        async with async_session_factory() as session:
            checkout = (
                await session.execute(
                    select(GoldEvolution).where(
                        GoldEvolution.tenant == tenant, GoldEvolution.canonical_name == "Checkout Service"
                    )
                )
            ).scalars().all()
        by_version = {row.version: row for row in checkout}
        assert set(by_version) == {1, 2}
        assert by_version[2].operation == "modified"  # corrected — the LLM said "unchanged"

        async with async_session_factory() as session:
            contract = (
                await session.execute(
                    select(GoldEvolution).where(
                        GoldEvolution.tenant == tenant, GoldEvolution.canonical_name == "checkout-completed"
                    )
                )
            ).scalars().one()
        output_ids = ComponentPayload.model_validate(by_version[2].payload).output_contract_ids
        assert output_ids == [contract.entity_id]
    finally:
        async with async_session_factory() as session:
            await session.execute(GoldEvolution.__table__.delete().where(GoldEvolution.tenant == tenant))
            await session.execute(GoldAlias.__table__.delete().where(GoldAlias.tenant == tenant))
            await session.commit()


async def test_current_gold_state_returns_only_the_latest_version_per_entity():
    entity_type = "component"
    entity_id = str(uuid4())
    base_kwargs = dict(
        entity_type=entity_type,
        entity_id=entity_id,
        canonical_name="Checkout Service",
        payload={"dependency_ids": [], "contract_ids": []},
        source_component="meeting.en.vtt",
        ingestion_date=date(2026, 6, 1),
    )
    try:
        async with async_session_factory() as session:
            await persist_entity_version(
                session, operation="new", narrative="v1 narrative", source_adr_version=1, **base_kwargs
            )
            await session.commit()
        async with async_session_factory() as session:
            await persist_entity_version(
                session, operation="modified", narrative="v2 narrative", source_adr_version=2, **base_kwargs
            )
            await session.commit()

        async with async_session_factory() as session:
            rows = await current_gold_state(session, entity_type=entity_type)
        matching = [r for r in rows if r.entity_id == entity_id]
        assert len(matching) == 1
        assert matching[0].version == 2
        assert matching[0].narrative == "v2 narrative"
    finally:
        await _cleanup_entity(entity_type, entity_id)


async def test_current_architecture_diagram_draws_a_node_per_live_component():
    tenant = f"test-arch-{uuid4().hex[:8]}"
    checkout_id, payment_id = str(uuid4()), str(uuid4())
    try:
        async with async_session_factory() as session:
            await persist_entity_version(
                session,
                entity_type="component",
                entity_id=payment_id,
                canonical_name="Payment Gateway",
                operation="new",
                narrative="Processes card payments.",
                payload={"dependency_ids": [], "contract_ids": []},
                source_component="meeting.en.vtt",
                source_adr_version=1,
                ingestion_date=date(2026, 6, 1),
                tenant=tenant,
            )
            await persist_entity_version(
                session,
                entity_type="component",
                entity_id=checkout_id,
                canonical_name="Checkout Service",
                operation="modified",
                narrative="Calls the payment gateway.",
                payload={"dependency_ids": [payment_id], "contract_ids": []},
                source_component="meeting.en.vtt",
                source_adr_version=2,
                ingestion_date=date(2026, 6, 2),
                tenant=tenant,
            )
            await session.commit()

        async with async_session_factory() as session:
            diagram = await current_architecture_diagram(session, tenant=tenant)

        assert diagram.startswith("flowchart LR")
        assert "classDef goldNode" in diagram
        # We use backtick markdown-string labels, never raw <br/>/<sub> HTML. See the
        # function's own docstring for why. Those HTML tags broke node visibility under
        # Streamlit's strict security mode.
        assert "<br/>" not in diagram and "<sub>" not in diagram
        assert '["`Payment Gateway\n*meeting v1*`"]' in diagram
        assert '["`Checkout Service\n*meeting v2*`"]' in diagram
        assert "-->" in diagram  # Checkout Service depends on Payment Gateway
        # There is no `click ... href` directive here. We confirmed, with a real headless-Chrome
        # run against `st.mermaid_chart` and not just mermaid-cli, that this directive makes
        # Streamlit render an empty node group while still drawing edges. That is exactly the
        # "only edges" bug. ADR navigation happens through the plain link table that
        # `_architecture_history_tab` renders instead.
        assert "click" not in diagram
        assert "class n0,n1 goldNode" in diagram
    finally:
        await _cleanup_entity("component", checkout_id)
        await _cleanup_entity("component", payment_id)


async def test_current_architecture_diagram_is_empty_when_gold_has_no_live_component():
    tenant = f"test-arch-empty-{uuid4().hex[:8]}"
    async with async_session_factory() as session:
        diagram = await current_architecture_diagram(session, tenant=tenant)
    assert diagram == ""


async def test_current_architecture_diagram_drops_a_removed_component():
    tenant = f"test-arch-{uuid4().hex[:8]}"
    entity_id = str(uuid4())
    try:
        async with async_session_factory() as session:
            await persist_entity_version(
                session,
                entity_type="component",
                entity_id=entity_id,
                canonical_name="Legacy Notifier",
                operation="removed",
                narrative="Decommissioned.",
                payload={"dependency_ids": [], "contract_ids": []},
                source_component="meeting.en.vtt",
                source_adr_version=1,
                ingestion_date=date(2026, 6, 1),
                tenant=tenant,
            )
            await session.commit()

        async with async_session_factory() as session:
            diagram = await current_architecture_diagram(session, tenant=tenant)
        assert diagram == ""
    finally:
        await _cleanup_entity("component", entity_id)


async def test_persist_entity_version_stores_authored_by():
    """`authored_by` copies forward the same value `silver_documents.authored_by` already
    holds — see `db/models.py`'s `GoldEvolution.authored_by` column comment. This is what lets
    an evolution timeline answer say who introduced or changed an entity."""
    entity_type = "component"
    entity_id = str(uuid4())
    try:
        async with async_session_factory() as session:
            await persist_entity_version(
                session,
                entity_type=entity_type,
                entity_id=entity_id,
                canonical_name="Checkout Service",
                operation="new",
                narrative="Checkout Service is new.",
                payload={"dependency_ids": [], "contract_ids": []},
                source_component="meeting.en.vtt",
                source_adr_version=1,
                ingestion_date=date(2026, 6, 1),
                authored_by="alex",
            )
            await session.commit()

        async with async_session_factory() as session:
            row = (
                await session.execute(
                    select(GoldEvolution).where(
                        GoldEvolution.entity_type == entity_type, GoldEvolution.entity_id == entity_id
                    )
                )
            ).scalar_one()
        assert row.authored_by == "alex"
    finally:
        await _cleanup_entity(entity_type, entity_id)


async def test_persist_entity_version_defaults_authored_by_to_empty_string():
    """A CLI, script, or test run with no real logged-in user must not crash or store `None` —
    the same `""` default `silver_documents.authored_by` and `bronze_documents.uploaded_by`
    already use."""
    entity_type = "component"
    entity_id = str(uuid4())
    try:
        async with async_session_factory() as session:
            await persist_entity_version(
                session,
                entity_type=entity_type,
                entity_id=entity_id,
                canonical_name="Checkout Service",
                operation="new",
                narrative="Checkout Service is new.",
                payload={"dependency_ids": [], "contract_ids": []},
                source_component="meeting.en.vtt",
                source_adr_version=1,
                ingestion_date=date(2026, 6, 1),
            )
            await session.commit()

        async with async_session_factory() as session:
            row = (
                await session.execute(
                    select(GoldEvolution).where(
                        GoldEvolution.entity_type == entity_type, GoldEvolution.entity_id == entity_id
                    )
                )
            ).scalar_one()
        assert row.authored_by == ""
    finally:
        await _cleanup_entity(entity_type, entity_id)


async def test_resolve_entity_id_for_lookup_returns_none_when_nothing_matches():
    """Unlike `resolve_entity_id`, the read-only lookup must never mint a fresh id for "no
    match" — a caller asking "does this entity exist" needs an honest `None`, not an empty,
    alias-less identity that would silently read as "found it, but no history"."""
    name = _unique_component_name()
    async with async_session_factory() as session:
        resolved = await resolve_entity_id_for_lookup(session, "component", name)
    assert resolved is None


async def test_resolve_entity_id_for_lookup_returns_existing_id_on_exact_match():
    entity_type = "component"
    entity_id = str(uuid4())
    name = _unique_component_name()
    try:
        async with async_session_factory() as session:
            await ensure_alias(session, entity_type, entity_id, name, "meeting.en.vtt", 1)
            await session.commit()

        async with async_session_factory() as session:
            resolved = await resolve_entity_id_for_lookup(session, entity_type, name)
        assert resolved == entity_id
    finally:
        await _cleanup_entity(entity_type, entity_id)


async def test_entity_history_returns_every_version_oldest_first():
    entity_type = "component"
    entity_id = str(uuid4())
    base_kwargs = dict(
        entity_type=entity_type,
        entity_id=entity_id,
        canonical_name="Checkout Service",
        payload={"dependency_ids": [], "contract_ids": []},
        source_component="meeting.en.vtt",
    )
    try:
        async with async_session_factory() as session:
            await persist_entity_version(
                session,
                operation="new",
                narrative="Introduced at launch.",
                source_adr_version=1,
                ingestion_date=date(2026, 6, 1),
                authored_by="alex",
                **base_kwargs,
            )
            await session.commit()
        async with async_session_factory() as session:
            await persist_entity_version(
                session,
                operation="modified",
                narrative="Added payment validation.",
                source_adr_version=2,
                ingestion_date=date(2026, 7, 1),
                authored_by="sam",
                **base_kwargs,
            )
            await session.commit()

        async with async_session_factory() as session:
            history = await entity_history(session, entity_type, entity_id)
        assert [row.version for row in history] == [1, 2]  # oldest first, not newest first
        assert history[0].authored_by == "alex"
        assert history[1].authored_by == "sam"
    finally:
        await _cleanup_entity(entity_type, entity_id)


async def test_entity_history_is_empty_for_an_entity_never_seen():
    async with async_session_factory() as session:
        history = await entity_history(session, "component", str(uuid4()))
    assert history == []


async def test_find_entity_by_name_in_text_matches_a_known_alias():
    entity_type = "component"
    entity_id = str(uuid4())
    name = f"Risk Notification Service {uuid4().hex[:8]}"
    try:
        async with async_session_factory() as session:
            await ensure_alias(session, entity_type, entity_id, name, "meeting.en.vtt", 1)
            await session.commit()

        async with async_session_factory() as session:
            match = await find_entity_by_name_in_text(
                session, f"What evolution has {name} had over time?"
            )
        assert match == (entity_type, entity_id, name)
    finally:
        await _cleanup_entity(entity_type, entity_id)


async def test_find_entity_by_name_in_text_returns_none_when_no_alias_appears():
    async with async_session_factory() as session:
        match = await find_entity_by_name_in_text(session, "What is the weather like today?")
    assert match is None


async def test_find_entity_by_name_in_text_prefers_the_longest_match():
    """"Order Service" and "Order" both being aliases (for two different entities) is a real
    scenario, not a contrived one: a short, generic fragment can end up registered as its own
    alias variant. The longer, more specific match wins — see the function's own docstring for
    why a short alias is too likely to be a coincidence."""
    short_id, long_id = str(uuid4()), str(uuid4())
    suffix = uuid4().hex[:8]
    short_name = f"Order{suffix}"
    long_name = f"Order{suffix} Fulfillment Service"
    try:
        async with async_session_factory() as session:
            await ensure_alias(session, "component", short_id, short_name, "meeting.en.vtt", 1)
            await ensure_alias(session, "component", long_id, long_name, "meeting.en.vtt", 1)
            await session.commit()

        async with async_session_factory() as session:
            match = await find_entity_by_name_in_text(
                session, f"How has {long_name} evolved over time?"
            )
        assert match == ("component", long_id, long_name)
    finally:
        await _cleanup_entity("component", short_id)
        await _cleanup_entity("component", long_id)


def test_is_evolution_question_recognizes_english_and_spanish_phrasing():
    assert is_evolution_question("How has the Order Service evolved over time?")
    assert is_evolution_question("What is the history of the checkout component?")
    assert is_evolution_question("¿Qué evolución ha tenido el componente X a lo largo del tiempo?")
    assert is_evolution_question("Dame la historia del componente de pagos.")


def test_is_evolution_question_is_false_for_an_ordinary_factual_question():
    assert not is_evolution_question("Who is the producer of the OrderCreated contract?")
    assert not is_evolution_question("What is the current status of the Checkout Service?")


# =====================================================================================
# Hybrid search (vector + lexical, fused with RRF) — see `.tmp/advanced_techniques.md` §1
# =====================================================================================


def test_reciprocal_rank_fusion_lets_a_strong_lexical_match_outrank_a_lone_vector_top_hit():
    """id 7 ranks only 7th in the vector ranking, but 1st (and only) in the lexical one.
    `1/(60+7) + 1/(60+1) ≈ 0.0313` beats id 1's `1/(60+1) ≈ 0.0164` — a lone rank-1 hit in one
    signal, with zero support from the other. This is the concrete case hybrid search exists
    for: a moderate embedding match that is ALSO the best lexical match should outrank a
    perfect embedding match that no lexical signal agrees with."""
    vector_ranking = list(range(1, 11))  # id 7 sits at position 7
    lexical_ranking = [7]
    fused = _reciprocal_rank_fusion([vector_ranking, lexical_ranking])
    assert fused[0] == 7


def test_reciprocal_rank_fusion_id_absent_from_a_ranking_gets_no_penalty_from_it():
    """An id missing from one input ranking contributes nothing from that ranking — never a
    negative score. Two ids that each appear in exactly one ranking, both at rank 1, must tie
    (both score `1/(60+1)`), not have the "missing" one penalized below a real rank."""
    fused = _reciprocal_rank_fusion([[1], [2]])
    assert set(fused) == {1, 2}


def test_reciprocal_rank_fusion_returns_empty_for_no_candidates():
    assert _reciprocal_rank_fusion([]) == []
    assert _reciprocal_rank_fusion([[], []]) == []


async def test_top_k_gold_evolution_hybrid_mode_finds_a_lexical_match_a_bad_vector_search_misses():
    """This proves the lexical branch actually rescues a row the vector branch alone would
    never surface — not just that the function runs. The query vector is deliberately set to
    the EXACT embedding of an unrelated distractor row (cosine distance 0), so `mode="vector"`
    is guaranteed to return only that distractor at `k=1`. `mode="hybrid"`, given the SAME
    adversarial vector but the real question text, must still surface the target row through
    `ts_rank` + RRF."""
    target_id, distractor_id = str(uuid4()), str(uuid4())
    unique_token = f"xzqv{uuid4().hex[:10]}"
    adversarial_vector = await embed_question("Completely unrelated text about weather forecasting.")
    try:
        async with async_session_factory() as session:
            await persist_entity_version(
                session,
                entity_type="component",
                entity_id=target_id,
                canonical_name=f"{unique_token} Adapter",
                operation="new",
                narrative=f"{unique_token} is a proprietary internal protocol adapter.",
                payload={"dependency_ids": [], "contract_ids": []},
                source_component="meeting.en.vtt",
                source_adr_version=1,
                ingestion_date=date(2026, 6, 1),
            )
            # A hand-built row whose embedding is IDENTICAL to the query vector — guaranteed
            # cosine distance 0, so it always wins rank 1 under mode="vector", regardless of
            # whatever else exists in this test database.
            session.add(
                GoldEvolution(
                    tenant="default",
                    entity_type="component",
                    entity_id=distractor_id,
                    canonical_name="Weather Distractor",
                    version=1,
                    operation="new",
                    narrative="Completely unrelated text about weather forecasting.",
                    payload={},
                    embedding=adversarial_vector,
                    entity_hash="distractor",
                    source_component="meeting.en.vtt",
                    source_adr_version=1,
                    ingestion_date=date(2026, 6, 1),
                )
            )
            await session.commit()

        async with async_session_factory() as session:
            vector_only = await top_k_gold_evolution(session, adversarial_vector, k=1, mode="vector")
            hybrid = await top_k_gold_evolution(
                session, adversarial_vector, k=5, mode="hybrid", question_text=f"What is {unique_token}?"
            )
        assert {r.entity_id for r in vector_only} == {distractor_id}
        assert target_id in {r.entity_id for r in hybrid}
    finally:
        await _cleanup_entity("component", target_id)
        await _cleanup_entity("component", distractor_id)


async def test_top_k_gold_evolution_hybrid_mode_requires_question_text():
    # `session=None`: both validation errors below raise before the function ever touches
    # `session`, so no real database session is needed to prove them.
    with pytest.raises(ValueError, match="question_text"):
        await top_k_gold_evolution(None, [0.0], mode="hybrid")


async def test_top_k_gold_evolution_rejects_an_unknown_mode():
    with pytest.raises(ValueError, match="unknown"):
        await top_k_gold_evolution(None, [0.0], mode="bm25")


# =====================================================================================
# Deduplication by entity in top-k — see `.tmp/advanced_techniques.md` §2
# =====================================================================================


async def test_dedupe_ids_by_entity_keeps_the_first_occurrence_per_entity_and_stops_at_k():
    """`_dedupe_ids_by_entity` trusts the caller's own `ids` order completely — it never
    re-ranks, it only groups by `(entity_type, entity_id)` and keeps the first (best-ranked) id
    it sees for each. Entity A's two versions collapse to just its first (best-ranked) one;
    entity B, seen once, is kept; the result stops as soon as `k` distinct entities are found,
    even though more ids remain in the input."""
    entity_a, entity_b, entity_c = str(uuid4()), str(uuid4()), str(uuid4())
    try:
        async with async_session_factory() as session:
            for version in (1, 2):
                await persist_entity_version(
                    session,
                    entity_type="component",
                    entity_id=entity_a,
                    canonical_name="Repeated Entity",
                    operation="new" if version == 1 else "modified",
                    narrative=f"version {version}",
                    payload={"dependency_ids": [], "contract_ids": []},
                    source_component="meeting.en.vtt",
                    source_adr_version=version,
                    ingestion_date=date(2026, 6, 1),
                )
            await persist_entity_version(
                session,
                entity_type="component",
                entity_id=entity_b,
                canonical_name="Other Entity",
                operation="new",
                narrative="b",
                payload={"dependency_ids": [], "contract_ids": []},
                source_component="meeting.en.vtt",
                source_adr_version=1,
                ingestion_date=date(2026, 6, 1),
            )
            await persist_entity_version(
                session,
                entity_type="component",
                entity_id=entity_c,
                canonical_name="Third Entity, never reached",
                operation="new",
                narrative="c",
                payload={"dependency_ids": [], "contract_ids": []},
                source_component="meeting.en.vtt",
                source_adr_version=1,
                ingestion_date=date(2026, 6, 1),
            )
            await session.commit()

        async with async_session_factory() as session:
            rows = (
                await session.execute(
                    select(GoldEvolution).where(
                        GoldEvolution.entity_id.in_([entity_a, entity_b, entity_c])
                    )
                )
            ).scalars().all()
            by_entity_and_version = {(r.entity_id, r.version): r.id for r in rows}
            a_v1_id = by_entity_and_version[(entity_a, 1)]
            a_v2_id = by_entity_and_version[(entity_a, 2)]
            b_id = by_entity_and_version[(entity_b, 1)]
            c_id = by_entity_and_version[(entity_c, 1)]

            # Deliberately "ranked" order: A's WORSE version first, to prove the function keeps
            # whichever occurrence comes first in `ids`, not necessarily version 1.
            ranked_ids = [a_v2_id, a_v1_id, b_id, c_id]
            deduped = await _dedupe_ids_by_entity(session, ranked_ids, k=2)
        assert deduped == [a_v2_id, b_id]  # A's first-seen version kept; C never reached (k=2)
    finally:
        await _cleanup_entity("component", entity_a)
        await _cleanup_entity("component", entity_b)
        await _cleanup_entity("component", entity_c)


async def test_dedupe_ids_by_entity_returns_empty_for_no_ids():
    async with async_session_factory() as session:
        assert await _dedupe_ids_by_entity(session, [], k=5) == []


async def test_top_k_gold_evolution_dedupe_rescues_a_different_entity_from_being_crowded_out():
    """This is the concrete bug dedup fixes. Three versions of the SAME entity, all persisted
    with the EXACT query embedding (distance 0 — unambiguously the closest possible match, no
    tie-breaking luck involved), fill every slot of a plain `k=3` top-k. A different, real
    entity — a nonzero but still close distance — never gets a chance to appear. With dedup
    (the default), the same query keeps only the repeated entity's single best-ranked version
    and still surfaces the other one."""
    query_vector = await embed_question("Anchor question used only to fix a query vector.")
    entity_a, entity_b = str(uuid4()), str(uuid4())
    try:
        async with async_session_factory() as session:
            for version in (1, 2, 3):
                session.add(
                    GoldEvolution(
                        tenant="default",
                        entity_type="component",
                        entity_id=entity_a,
                        canonical_name="Repeated Entity",
                        version=version,
                        operation="new" if version == 1 else "modified",
                        narrative=f"Repeated Entity narrative, version {version}.",
                        payload={},
                        embedding=query_vector,  # distance 0 to the query, every single version
                        entity_hash=f"hash-a-{version}",
                        source_component="meeting.en.vtt",
                        source_adr_version=version,
                        ingestion_date=date(2026, 6, 1),
                    )
                )
            session.add(
                GoldEvolution(
                    tenant="default",
                    entity_type="component",
                    entity_id=entity_b,
                    canonical_name="Other Entity",
                    version=1,
                    operation="new",
                    narrative="A different, unrelated component.",
                    payload={},
                    embedding=await embed_question("Something else entirely, unrelated to the anchor."),
                    entity_hash="hash-b",
                    source_component="meeting.en.vtt",
                    source_adr_version=1,
                    ingestion_date=date(2026, 6, 1),
                )
            )
            await session.commit()

        async with async_session_factory() as session:
            without_dedupe = await top_k_gold_evolution(session, query_vector, k=3, dedupe=False)
            with_dedupe = await top_k_gold_evolution(session, query_vector, k=3, dedupe=True)

        assert {r.entity_id for r in without_dedupe} == {entity_a}  # entity_b crowded out entirely
        assert {entity_a, entity_b} <= {r.entity_id for r in with_dedupe}  # both rescued
    finally:
        await _cleanup_entity("component", entity_a)
        await _cleanup_entity("component", entity_b)


async def test_dedupe_ids_by_entity_resolves_to_the_latest_version_not_the_better_ranked_one():
    """Regression test for a real bug `agents.stages.gold.testing`'s real-LLM suite caught: an
    entity's OLDER version can rank ahead of its own current state (its narrative happens to
    word-match the ranking signal more closely). The first implementation of dedup kept
    whichever version ranked best, which meant a "what changed in this latest update?"
    question got answered from a stale, superseded version — missing the very update it asked
    about. Dedup must always resolve to the entity's actual latest version, regardless of which
    version's id ranked better in the input list."""
    entity_a = str(uuid4())
    try:
        async with async_session_factory() as session:
            for version in (1, 2):
                await persist_entity_version(
                    session,
                    entity_type="component",
                    entity_id=entity_a,
                    canonical_name="Order Service",
                    operation="new" if version == 1 else "modified",
                    narrative=f"version {version} narrative",
                    payload={"dependency_ids": [], "contract_ids": []},
                    source_component="meeting.en.vtt",
                    source_adr_version=version,
                    ingestion_date=date(2026, 6, 1),
                )
            await session.commit()

        async with async_session_factory() as session:
            rows = (
                await session.execute(select(GoldEvolution).where(GoldEvolution.entity_id == entity_a))
            ).scalars().all()
            by_version = {r.version: r.id for r in rows}
            v1_id, v2_id = by_version[1], by_version[2]

            # v1, the OLDER version, is the only one in the ranked input — simulating exactly
            # the observed failure: an older narrative ranked inside the recall pool while the
            # entity's actual latest version did not (or ranked worse).
            deduped = await _dedupe_ids_by_entity(session, [v1_id], k=1)
        assert deduped == [v2_id]
    finally:
        await _cleanup_entity("component", entity_a)


# =====================================================================================
# Reranking (cross-encoder) — see `.tmp/advanced_techniques.md` §3
# =====================================================================================


def test_score_candidates_ranks_a_relevant_document_above_an_irrelevant_one():
    """A basic sanity check on the cross-encoder itself, isolated from anything Gold-specific:
    given one document that actually answers the query and one that has nothing to do with
    it, the relevant one must score higher."""
    scores = score_candidates(
        "What is the capital of France?",
        ["Paris is the capital of France.", "Bananas are a good source of potassium."],
    )
    assert scores[0] > scores[1]


async def test_rerank_ids_reorders_by_relevance_not_by_input_order():
    """`_rerank_ids` must produce a NEW ranking from the actual (question, narrative) pairs —
    not just echo back whatever order `ids` arrived in. The irrelevant row is listed FIRST in
    the input `ids`; a correct rerank must still put the relevant one first."""
    relevant_id, irrelevant_id = str(uuid4()), str(uuid4())
    try:
        async with async_session_factory() as session:
            await persist_entity_version(
                session,
                entity_type="component",
                entity_id=irrelevant_id,
                canonical_name="Irrelevant Component",
                operation="new",
                narrative="This component has nothing to do with payment processing at all.",
                payload={"dependency_ids": [], "contract_ids": []},
                source_component="meeting.en.vtt",
                source_adr_version=1,
                ingestion_date=date(2026, 6, 1),
            )
            await persist_entity_version(
                session,
                entity_type="component",
                entity_id=relevant_id,
                canonical_name="Payments Gateway",
                operation="new",
                narrative="The Payments Gateway processes card payments and refunds for checkout.",
                payload={"dependency_ids": [], "contract_ids": []},
                source_component="meeting.en.vtt",
                source_adr_version=1,
                ingestion_date=date(2026, 6, 1),
            )
            await session.commit()

        async with async_session_factory() as session:
            rows = (
                await session.execute(
                    select(GoldEvolution).where(GoldEvolution.entity_id.in_([relevant_id, irrelevant_id]))
                )
            ).scalars().all()
            id_by_entity = {r.entity_id: r.id for r in rows}
            irrelevant_row_id, relevant_row_id = id_by_entity[irrelevant_id], id_by_entity[relevant_id]

            # Deliberately wrong input order: the irrelevant row listed first.
            reranked = await _rerank_ids(
                session, "How does this system process card payments?", [irrelevant_row_id, relevant_row_id], k=2
            )
        assert reranked == [relevant_row_id, irrelevant_row_id]
    finally:
        await _cleanup_entity("component", relevant_id)
        await _cleanup_entity("component", irrelevant_id)


async def test_rerank_ids_returns_empty_for_no_ids():
    async with async_session_factory() as session:
        assert await _rerank_ids(session, "any question", [], k=5) == []


async def test_top_k_gold_evolution_rerank_true_requires_question_text():
    with pytest.raises(ValueError, match="question_text"):
        await top_k_gold_evolution(None, [0.0], rerank=True)


async def test_top_k_gold_evolution_rerank_promotes_a_semantically_relevant_row():
    """End-to-end proof that `rerank=True` changes the final top-k, not just that `_rerank_ids`
    works in isolation. The query embedding is set to the IRRELEVANT row's own embedding
    (distance 0 to it, guaranteed rank 1 by plain vector search), while the relevant row gets
    an unrelated embedding (guaranteed to rank worse). Plain vector search (`rerank=False`)
    must then return the irrelevant row first; `rerank=True`, given the real question text,
    must promote the actually-relevant row instead."""
    relevant_id, irrelevant_id = str(uuid4()), str(uuid4())
    question = "How does this system process card payments?"
    irrelevant_embedding = await embed_question("This narrative's own embedding, unrelated to the question.")
    try:
        async with async_session_factory() as session:
            session.add(
                GoldEvolution(
                    tenant="default",
                    entity_type="component",
                    entity_id=irrelevant_id,
                    canonical_name="Irrelevant Component",
                    version=1,
                    operation="new",
                    narrative="This component has nothing to do with payment processing at all.",
                    payload={},
                    embedding=irrelevant_embedding,  # distance 0 to the query vector below
                    entity_hash="hash-irrelevant",
                    source_component="meeting.en.vtt",
                    source_adr_version=1,
                    ingestion_date=date(2026, 6, 1),
                )
            )
            session.add(
                GoldEvolution(
                    tenant="default",
                    entity_type="component",
                    entity_id=relevant_id,
                    canonical_name="Payments Gateway",
                    version=1,
                    operation="new",
                    narrative="The Payments Gateway processes card payments and refunds for checkout.",
                    payload={},
                    embedding=await embed_question("A third, unrelated anchor text."),
                    entity_hash="hash-relevant",
                    source_component="meeting.en.vtt",
                    source_adr_version=1,
                    ingestion_date=date(2026, 6, 1),
                )
            )
            await session.commit()

        async with async_session_factory() as session:
            without_rerank = await top_k_gold_evolution(session, irrelevant_embedding, k=1, rerank=False)
            with_rerank = await top_k_gold_evolution(
                session, irrelevant_embedding, k=1, rerank=True, question_text=question
            )
        assert {r.entity_id for r in without_rerank} == {irrelevant_id}
        assert {r.entity_id for r in with_rerank} == {relevant_id}
    finally:
        await _cleanup_entity("component", relevant_id)
        await _cleanup_entity("component", irrelevant_id)
