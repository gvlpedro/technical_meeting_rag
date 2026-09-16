"""Gold layer storage — `agents/gold_service.py` against real Postgres, mirroring
`tests/test_silver_storage.py`'s own style. No LLM involved anywhere in this file: identity
resolution, hashing, and versioning are all deterministic, DB-only logic (`gold_process.md`
§3/§5) — `tests/test_clarification_loop.py`'s Gold-extension test covers the graph-level,
faked-LLM path end to end.
"""

from datetime import date
from uuid import uuid4

import pytest
from sqlalchemy import delete, select

from agents.gold_service import (
    already_extracted,
    current_architecture_diagram,
    current_gold_state,
    ensure_alias,
    persist_entity_version,
    resolve_entity_id,
)
from agents.schemas import ArchitecturePayload, ComponentPayload, DataContractPayload
from db.models import GoldAlias, GoldEvolution
from db.session import async_session_factory

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _unique_component_name() -> str:
    # Real names, not uuids, so word-boundary-shaped inputs stay realistic — just made
    # collision-free per test run since gold_aliases has no per-test isolation column to
    # scope a WHERE clause on the way bronze/silver tests scope by ingestion_date.
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
    assert payload.model_dump() == {"dependency_ids": ["e1", "e2"], "contract_ids": ["c1"]}


def test_component_payload_defaults_to_empty_lists():
    assert ComponentPayload().model_dump() == {"dependency_ids": [], "contract_ids": []}


def test_data_contract_payload_keeps_odcs_spec_opaque():
    spec = {"apiVersion": "v3.1.0", "schema": {"type": "object", "properties": {"a": {"type": "string"}}}}
    payload = DataContractPayload(producer="checkout", consumer="billing", odcs_spec=spec)
    assert payload.model_dump()["odcs_spec"] == spec  # not destructured/validated, stored as given


def test_architecture_payload_defaults():
    assert ArchitecturePayload().model_dump() == {
        "mermaid_diagram": "",
        "components": [],
        "dependencies": [],
    }


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
    misspelled = canonical.replace("Checkout", "Checkot")  # one typo, still >0.6 trigram similarity
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
    assert resolved  # a fresh uuid4 string, not empty
    # Never inserted anywhere — resolve_entity_id only resolves, ensure_alias inserts.
    async with async_session_factory() as session:
        rows = await session.execute(
            select(GoldAlias).where(GoldAlias.entity_id == resolved)
        )
        assert rows.first() is None


async def test_ensure_alias_is_idempotent():
    entity_type = "component"
    entity_id = str(uuid4())
    name = _unique_component_name()
    try:
        async with async_session_factory() as session:
            await ensure_alias(session, entity_type, entity_id, name, "meeting.en.vtt", 1)
            await ensure_alias(session, entity_type, entity_id, name, "meeting.en.vtt", 1)  # same triple twice
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
        assert second is None  # identical (operation, narrative, payload) -> no new row

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
                    select(GoldEvolution).where(
                        GoldEvolution.entity_type == entity_type, GoldEvolution.entity_id == entity_id
                    )
                )
            ).all()
        assert len(rows) == 2  # both versions kept, older row untouched
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


async def test_current_architecture_diagram_draws_a_node_per_live_component_with_an_adr_link():
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
        # Backtick markdown-string labels, never raw <br/>/<sub> HTML (see the function's own
        # docstring for why: those broke node visibility under Streamlit's strict security mode).
        assert "<br/>" not in diagram and "<sub>" not in diagram
        assert '["`Payment Gateway\n*meeting v1*`"]' in diagram
        assert '["`Checkout Service\n*meeting v2*`"]' in diagram
        assert "-->" in diagram  # Checkout Service depends on Payment Gateway
        assert 'click' in diagram and '?view_adr=meeting.en.vtt&view_adr_version=1' in diagram
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
