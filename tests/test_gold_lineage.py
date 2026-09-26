"""Covers `.tmp/improve_timeline_questions_and_linage.md`: input/output contract direction,
`odcs_spec` carry-forward on "unchanged" contracts, vigencia temporal (`valid_to`/`as_of`), and
predecessors/successors. `classify_contract_directions` gets its own fast, no-DB unit tests; the
rest need real Postgres since they exercise `persist_entity_version`'s hash-compare-then-bump and
`valid_to` closing directly."""

from datetime import date
from uuid import uuid4

import pytest

from agents.stages.gold.service import (
    classify_contract_directions,
    current_gold_state_as_of,
    get_component_contracts,
    get_predecessors,
    get_successors,
    latest_odcs_spec,
    persist_entity_version,
)
from db.models import GoldAlias, GoldEvolution
from db.session import async_session_factory

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


async def _cleanup(tenant: str) -> None:
    async with async_session_factory() as session:
        await session.execute(GoldEvolution.__table__.delete().where(GoldEvolution.tenant == tenant))
        await session.execute(GoldAlias.__table__.delete().where(GoldAlias.tenant == tenant))
        await session.commit()


# --- classify_contract_directions, in isolation: no DB ---------------------------------------


def test_classify_contract_directions_splits_by_producer_and_consumer():
    name_to_id = {"checkout-events": "contract-1", "Checkout Service": "comp-a", "Billing": "comp-b"}
    contracts = [
        {"name": "checkout-events", "action": "new", "producer": "Checkout Service", "consumer": "Billing"}
    ]
    directions = classify_contract_directions(contracts, name_to_id)
    assert directions["Checkout Service"] == {"input": set(), "output": {"contract-1"}}
    assert directions["Billing"] == {"input": {"contract-1"}, "output": set()}


def test_classify_contract_directions_skips_unknown_action_and_unresolved_names():
    name_to_id = {"Checkout Service": "comp-a", "Billing": "comp-b"}
    contracts = [
        {"name": "checkout-events", "action": "unknown", "producer": "Checkout Service", "consumer": "Billing"},
        {"name": "not-resolved", "action": "new", "producer": "Checkout Service", "consumer": "Billing"},
    ]
    assert classify_contract_directions(contracts, name_to_id) == {}


# --- latest_odcs_spec (carry-forward) ---------------------------------------------------------


async def test_latest_odcs_spec_is_empty_dict_when_never_persisted():
    async with async_session_factory() as session:
        assert await latest_odcs_spec(session, str(uuid4()), tenant="default") == {}


async def test_latest_odcs_spec_carries_forward_from_the_last_populated_version():
    tenant = f"lineage-{uuid4().hex[:8]}"
    entity_id = str(uuid4())
    try:
        spec = {"apiVersion": "v3.1.0", "schema": {"properties": {"name": {"type": "string"}}}}
        async with async_session_factory() as session:
            await persist_entity_version(
                session,
                entity_type="data_contract",
                entity_id=entity_id,
                canonical_name="products",
                operation="new",
                narrative="products contract defines the product schema.",
                payload={"producer": "Catalog", "consumer": "Storefront", "odcs_spec": spec},
                source_component="meeting-1.en.vtt",
                source_adr_version=1,
                ingestion_date=date(2026, 1, 1),
                tenant=tenant,
            )
            await session.commit()

        # A later ADR marks it "unchanged" and its own extraction naturally has no spec to give.
        async with async_session_factory() as session:
            carried = await latest_odcs_spec(session, entity_id, tenant=tenant)
        assert carried == spec  # This is what closes the real bug: it must not come back {}.
    finally:
        await _cleanup(tenant)


# --- input/output split + full spec, via get_component_contracts -----------------------------


async def test_get_component_contracts_splits_by_direction_with_full_spec():
    tenant = f"lineage-{uuid4().hex[:8]}"
    component_id, input_contract_id, output_contract_id = str(uuid4()), str(uuid4()), str(uuid4())
    try:
        async with async_session_factory() as session:
            await persist_entity_version(
                session, entity_type="component", entity_id=component_id, canonical_name="Checkout Service",
                operation="new", narrative="Checkout Service.",
                payload={
                    "dependency_ids": [], "contract_ids": [input_contract_id, output_contract_id],
                    "input_contract_ids": [input_contract_id], "output_contract_ids": [output_contract_id],
                },
                source_component="meeting-1.en.vtt", source_adr_version=1, ingestion_date=date(2026, 1, 1),
                tenant=tenant,
            )
            await persist_entity_version(
                session, entity_type="data_contract", entity_id=input_contract_id, canonical_name="payment-events",
                operation="new", narrative="payment-events contract.",
                payload={"producer": "Billing", "consumer": "Checkout Service", "odcs_spec": {"apiVersion": "v1"}},
                source_component="meeting-1.en.vtt", source_adr_version=1, ingestion_date=date(2026, 1, 1),
                tenant=tenant,
            )
            await persist_entity_version(
                session, entity_type="data_contract", entity_id=output_contract_id, canonical_name="checkout-events",
                operation="new", narrative="checkout-events contract.",
                payload={"producer": "Checkout Service", "consumer": "Billing", "odcs_spec": {"apiVersion": "v2"}},
                source_component="meeting-1.en.vtt", source_adr_version=1, ingestion_date=date(2026, 1, 1),
                tenant=tenant,
            )
            await session.commit()

        async with async_session_factory() as session:
            contracts = await get_component_contracts(session, component_id, tenant=tenant)

        assert [c.odcs_spec for c in contracts["input"]] == [{"apiVersion": "v1"}]
        assert [c.odcs_spec for c in contracts["output"]] == [{"apiVersion": "v2"}]
    finally:
        await _cleanup(tenant)


# --- predecessors / successors -----------------------------------------------------------------


async def test_get_successors_is_the_inverse_of_get_predecessors():
    tenant = f"lineage-{uuid4().hex[:8]}"
    upstream_id, downstream_a_id, downstream_b_id, unrelated_id = (
        str(uuid4()), str(uuid4()), str(uuid4()), str(uuid4())
    )
    try:
        async with async_session_factory() as session:
            await persist_entity_version(
                session, entity_type="component", entity_id=upstream_id, canonical_name="Payments Gateway",
                operation="new", narrative="Payments Gateway.", payload={"dependency_ids": [], "contract_ids": []},
                source_component="meeting-1.en.vtt", source_adr_version=1, ingestion_date=date(2026, 1, 1),
                tenant=tenant,
            )
            # Two components independently declare a dependency on `upstream_id` — no shared write.
            await persist_entity_version(
                session, entity_type="component", entity_id=downstream_a_id, canonical_name="Checkout Service",
                operation="new", narrative="Checkout Service.",
                payload={"dependency_ids": [upstream_id], "contract_ids": []},
                source_component="meeting-1.en.vtt", source_adr_version=1, ingestion_date=date(2026, 1, 1),
                tenant=tenant,
            )
            await persist_entity_version(
                session, entity_type="component", entity_id=downstream_b_id, canonical_name="Fraud Check Service",
                operation="new", narrative="Fraud Check Service.",
                payload={"dependency_ids": [upstream_id], "contract_ids": []},
                source_component="meeting-2.en.vtt", source_adr_version=1, ingestion_date=date(2026, 1, 2),
                tenant=tenant,
            )
            await persist_entity_version(
                session, entity_type="component", entity_id=unrelated_id, canonical_name="Reporting Service",
                operation="new", narrative="Reporting Service.", payload={"dependency_ids": [], "contract_ids": []},
                source_component="meeting-3.en.vtt", source_adr_version=1, ingestion_date=date(2026, 1, 3),
                tenant=tenant,
            )
            await session.commit()

        async with async_session_factory() as session:
            predecessors = await get_predecessors(session, downstream_a_id, tenant=tenant)
            successors = await get_successors(session, upstream_id, tenant=tenant)

        assert {r.entity_id for r in predecessors} == {upstream_id}
        assert {r.entity_id for r in successors} == {downstream_a_id, downstream_b_id}  # never `unrelated_id`
    finally:
        await _cleanup(tenant)


async def test_get_successors_as_of_excludes_an_edge_declared_later():
    tenant = f"lineage-{uuid4().hex[:8]}"
    upstream_id, downstream_id = str(uuid4()), str(uuid4())
    try:
        async with async_session_factory() as session:
            await persist_entity_version(
                session, entity_type="component", entity_id=upstream_id, canonical_name="Payments Gateway",
                operation="new", narrative="Payments Gateway.", payload={"dependency_ids": [], "contract_ids": []},
                source_component="meeting-1.en.vtt", source_adr_version=1, ingestion_date=date(2026, 1, 1),
                tenant=tenant,
            )
            # The dependency edge is only introduced on 2026-02-01, in a later ADR.
            await persist_entity_version(
                session, entity_type="component", entity_id=downstream_id, canonical_name="Checkout Service",
                operation="new", narrative="Checkout Service.", payload={"dependency_ids": [], "contract_ids": []},
                source_component="meeting-2.en.vtt", source_adr_version=1, ingestion_date=date(2026, 1, 10),
                tenant=tenant,
            )
            await persist_entity_version(
                session, entity_type="component", entity_id=downstream_id, canonical_name="Checkout Service",
                operation="modified", narrative="Checkout Service now depends on Payments Gateway.",
                payload={"dependency_ids": [upstream_id], "contract_ids": []},
                source_component="meeting-2.en.vtt", source_adr_version=2, ingestion_date=date(2026, 2, 1),
                tenant=tenant,
            )
            await session.commit()

        async with async_session_factory() as session:
            before = await get_successors(session, upstream_id, tenant=tenant, as_of=date(2026, 1, 15))
            after = await get_successors(session, upstream_id, tenant=tenant, as_of=date(2026, 2, 15))

        assert before == []  # The edge did not exist yet as of this date.
        assert {r.entity_id for r in after} == {downstream_id}
    finally:
        await _cleanup(tenant)


# --- current_gold_state_as_of -------------------------------------------------------------------


async def test_current_gold_state_as_of_returns_the_version_valid_on_that_date():
    tenant = f"lineage-{uuid4().hex[:8]}"
    entity_id = str(uuid4())
    try:
        async with async_session_factory() as session:
            await persist_entity_version(
                session, entity_type="component", entity_id=entity_id, canonical_name="Checkout Service",
                operation="new", narrative="v1 narrative.", payload={"dependency_ids": [], "contract_ids": []},
                source_component="meeting-1.en.vtt", source_adr_version=1, ingestion_date=date(2026, 1, 1),
                tenant=tenant,
            )
            await persist_entity_version(
                session, entity_type="component", entity_id=entity_id, canonical_name="Checkout Service",
                operation="modified", narrative="v2 narrative.", payload={"dependency_ids": [], "contract_ids": []},
                source_component="meeting-1.en.vtt", source_adr_version=2, ingestion_date=date(2026, 3, 1),
                tenant=tenant,
            )
            await session.commit()

        async with async_session_factory() as session:
            before_v1 = await current_gold_state_as_of(session, date(2025, 12, 1), "component", tenant=tenant)
            during_v1 = await current_gold_state_as_of(session, date(2026, 2, 1), "component", tenant=tenant)
            during_v2 = await current_gold_state_as_of(session, date(2026, 6, 1), "component", tenant=tenant)

        assert before_v1 == []  # Nothing existed yet.
        assert [r.narrative for r in during_v1] == ["v1 narrative."]
        assert [r.narrative for r in during_v2] == ["v2 narrative."]
    finally:
        await _cleanup(tenant)
