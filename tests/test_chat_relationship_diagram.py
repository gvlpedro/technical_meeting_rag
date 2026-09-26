"""Covers the "always show a small relationship diagram" feature: whenever a chat answer CITES
a component, `/chat` also returns a Mermaid `flowchart LR` of that component plus its direct
neighbors (predecessors via `dependency_ids`, successors via `get_successors`), scoped to only
the CITED component(s) — never a row that was merely retrieved but never used, and never a
neighbor's own ADR in `diagram_sources`. `build_relationship_diagram` gets unit tests against
real Postgres; `/chat`'s wiring gets an end-to-end test proving the scoping (cited vs. merely
retrieved) is honored."""

import json
from datetime import date
from types import SimpleNamespace
from uuid import uuid4

import litellm
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from agents.stages import gold
from agents.stages.gold.schemas import ComponentPayload, DataContractPayload
from agents.stages.gold.service import build_relationship_diagram, persist_entity_version
from app.config import settings
from app.main import app
from db.models import GoldAlias, GoldEvolution
from db.session import async_session_factory

pytestmark = pytest.mark.anyio
client = TestClient(app)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _fake_response(content: str) -> SimpleNamespace:
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice], model="fake")


async def _cleanup(tenant: str) -> None:
    async with async_session_factory() as session:
        await session.execute(GoldEvolution.__table__.delete().where(GoldEvolution.tenant == tenant))
        await session.execute(GoldAlias.__table__.delete().where(GoldAlias.tenant == tenant))
        await session.commit()


# --- build_relationship_diagram, against real Postgres ----------------------------------------


async def test_build_relationship_diagram_includes_the_focal_component_and_its_direct_neighbors():
    tenant = f"diagram-{uuid4().hex[:8]}"
    checkout_id, payments_id, unrelated_id = str(uuid4()), str(uuid4()), str(uuid4())
    try:
        async with async_session_factory() as session:
            await persist_entity_version(
                session, entity_type="component", entity_id=payments_id, canonical_name="Payments Gateway",
                operation="new", narrative="Payments Gateway.", payload=ComponentPayload().model_dump(),
                source_component="adr1.txt", source_adr_version=1, ingestion_date=date(2026, 1, 1), tenant=tenant,
            )
            await persist_entity_version(
                session, entity_type="component", entity_id=checkout_id, canonical_name="Checkout Service",
                operation="new", narrative="Checkout Service.",
                payload=ComponentPayload(dependency_ids=[payments_id]).model_dump(),
                source_component="adr2.txt", source_adr_version=1, ingestion_date=date(2026, 1, 2), tenant=tenant,
            )
            # A third, unrelated component — must never appear in a diagram scoped to Checkout Service.
            await persist_entity_version(
                session, entity_type="component", entity_id=unrelated_id, canonical_name="Reporting Service",
                operation="new", narrative="Reporting Service.", payload=ComponentPayload().model_dump(),
                source_component="adr3.txt", source_adr_version=1, ingestion_date=date(2026, 1, 3), tenant=tenant,
            )
            await session.commit()

            checkout_row = (
                await session.execute(select(GoldEvolution).where(GoldEvolution.entity_id == checkout_id))
            ).scalars().one()
            built = await build_relationship_diagram(session, [checkout_row], tenant=tenant)

        assert built is not None
        diagram, focal_rows = built
        assert [r.canonical_name for r in focal_rows] == ["Checkout Service"]
        assert "Checkout Service" in diagram
        assert "Payments Gateway" in diagram
        assert "Reporting Service" not in diagram
    finally:
        await _cleanup(tenant)


async def test_build_relationship_diagram_returns_none_for_a_component_with_no_relationships():
    tenant = f"diagram-{uuid4().hex[:8]}"
    lonely_id = str(uuid4())
    try:
        async with async_session_factory() as session:
            await persist_entity_version(
                session, entity_type="component", entity_id=lonely_id, canonical_name="Lonely Service",
                operation="new", narrative="Lonely Service.", payload=ComponentPayload().model_dump(),
                source_component="adr1.txt", source_adr_version=1, ingestion_date=date(2026, 1, 1), tenant=tenant,
            )
            await session.commit()
            row = (
                await session.execute(select(GoldEvolution).where(GoldEvolution.entity_id == lonely_id))
            ).scalars().one()
            assert await build_relationship_diagram(session, [row], tenant=tenant) is None
    finally:
        await _cleanup(tenant)


async def test_build_relationship_diagram_for_an_old_version_never_shows_a_later_added_relationship():
    """Regression for a real, reported bug: pinning an answer to an old version (`show me
    backend in v1`) still showed TODAY's full neighbor set, because neighbors/successors were
    resolved with no `as_of` at all. A v1 diagram must show only what was true as of v1's own
    `ingestion_date` — not a dependency or successor added in a later version."""
    tenant = f"diagram-{uuid4().hex[:8]}"
    backend_id, frontend_id, postgres_id = str(uuid4()), str(uuid4()), str(uuid4())
    try:
        async with async_session_factory() as session:
            # v1 (2026-01-01): backend depends on nothing yet; frontend depends on backend.
            await persist_entity_version(
                session, entity_type="component", entity_id=backend_id, canonical_name="backend",
                operation="new", narrative="backend v1.", payload=ComponentPayload().model_dump(),
                source_component="adr1.txt", source_adr_version=1, ingestion_date=date(2026, 1, 1), tenant=tenant,
            )
            await persist_entity_version(
                session, entity_type="component", entity_id=frontend_id, canonical_name="frontend",
                operation="new", narrative="frontend v1.",
                payload=ComponentPayload(dependency_ids=[backend_id]).model_dump(),
                source_component="adr1.txt", source_adr_version=1, ingestion_date=date(2026, 1, 1), tenant=tenant,
            )
            # v2 (2026-02-01): PostgreSQL is introduced, and backend is modified to depend on it.
            await persist_entity_version(
                session, entity_type="component", entity_id=postgres_id, canonical_name="PostgreSQL",
                operation="new", narrative="PostgreSQL introduced.", payload=ComponentPayload().model_dump(),
                source_component="adr2.txt", source_adr_version=1, ingestion_date=date(2026, 2, 1), tenant=tenant,
            )
            await persist_entity_version(
                session, entity_type="component", entity_id=backend_id, canonical_name="backend",
                operation="modified", narrative="backend v2 now uses PostgreSQL.",
                payload=ComponentPayload(dependency_ids=[postgres_id]).model_dump(),
                source_component="adr2.txt", source_adr_version=1, ingestion_date=date(2026, 2, 1), tenant=tenant,
            )
            await session.commit()

            v1_row = (
                await session.execute(
                    select(GoldEvolution).where(GoldEvolution.entity_id == backend_id, GoldEvolution.version == 1)
                )
            ).scalars().one()
            built = await build_relationship_diagram(session, [v1_row], tenant=tenant)

        assert built is not None
        diagram, _ = built
        assert "frontend" in diagram  # a real v1 relationship (frontend depended on backend then)
        assert "PostgreSQL" not in diagram  # only added in v2 — must not leak into the v1 diagram
    finally:
        await _cleanup(tenant)


async def test_build_relationship_diagram_returns_none_when_rows_has_no_component():
    async with async_session_factory() as session:
        contract_row = SimpleNamespace(entity_type="data_contract")
        assert await build_relationship_diagram(session, [contract_row], tenant="default") is None


# --- /chat wiring: diagram is scoped to CITED components only ---------------------------------


async def test_chat_diagram_is_scoped_to_the_cited_component_not_a_merely_retrieved_one(monkeypatch):
    tenant = settings.frontend_users[0].tenant
    checkout_id, payments_id, uncited_id = str(uuid4()), str(uuid4()), str(uuid4())
    try:
        async with async_session_factory() as session:
            await persist_entity_version(
                session, entity_type="component", entity_id=payments_id, canonical_name="Payments Gateway",
                operation="new", narrative="Payments Gateway.", payload=ComponentPayload().model_dump(),
                source_component="adr1.txt", source_adr_version=1, ingestion_date=date(2026, 1, 1), tenant=tenant,
            )
            await persist_entity_version(
                session, entity_type="component", entity_id=checkout_id, canonical_name="Checkout Service",
                operation="new", narrative="Checkout Service.",
                payload=ComponentPayload(dependency_ids=[payments_id]).model_dump(),
                source_component="adr2.txt", source_adr_version=1, ingestion_date=date(2026, 1, 2), tenant=tenant,
            )
            # Retrieved alongside Checkout Service, but never cited by the answer below — must
            # not leak into the diagram or its ADR footer.
            await persist_entity_version(
                session, entity_type="component", entity_id=uncited_id, canonical_name="Uncited Service",
                operation="new", narrative="Uncited Service.", payload=ComponentPayload().model_dump(),
                source_component="adr3.txt", source_adr_version=1, ingestion_date=date(2026, 1, 3), tenant=tenant,
            )
            await session.commit()

        async def fake_top_k(*args, **kwargs):
            async with async_session_factory() as session:
                return (
                    await session.execute(
                        select(GoldEvolution).where(GoldEvolution.entity_id.in_([checkout_id, uncited_id]))
                    )
                ).scalars().all()

        monkeypatch.setattr(gold, "top_k_gold_evolution", fake_top_k)

        async def fake_acompletion(*, model, api_key, messages, **kwargs):
            payload = {
                "answer": "Checkout Service depends on Payments Gateway.",
                "citations": [{"entity_type": "component", "entity_id": checkout_id, "version": 1}],
            }
            return _fake_response(json.dumps(payload))

        monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

        response = client.post(
            "/v1/frontend/chat",
            json={"username": settings.frontend_users[0].username, "question": "What does Checkout Service depend on?"},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["diagram"] is not None
        assert "Checkout Service" in body["diagram"]
        assert "Payments Gateway" in body["diagram"]
        assert "Uncited Service" not in body["diagram"]
        assert body["diagram_sources"] == [
            {"canonical_name": "Checkout Service", "source_component": "adr2.txt", "source_adr_version": 1}
        ]
    finally:
        await _cleanup(tenant)


async def test_chat_diagram_is_absent_when_the_answer_cites_no_component(monkeypatch):
    user = settings.frontend_users[0]
    contract_id = str(uuid4())
    try:
        async with async_session_factory() as session:
            await persist_entity_version(
                session, entity_type="data_contract", entity_id=contract_id, canonical_name="PaymentCharged",
                operation="new", narrative="PaymentCharged event.",
                payload=DataContractPayload(producer="x", consumer="y").model_dump(),
                source_component="adr1.txt", source_adr_version=1, ingestion_date=date(2026, 1, 1), tenant=user.tenant,
            )
            await session.commit()

        async def fake_top_k(*args, **kwargs):
            async with async_session_factory() as session:
                return (
                    await session.execute(select(GoldEvolution).where(GoldEvolution.entity_id == contract_id))
                ).scalars().all()

        monkeypatch.setattr(gold, "top_k_gold_evolution", fake_top_k)

        async def fake_acompletion(*, model, api_key, messages, **kwargs):
            payload = {
                "answer": "PaymentCharged is a data contract.",
                "citations": [{"entity_type": "data_contract", "entity_id": contract_id, "version": 1}],
            }
            return _fake_response(json.dumps(payload))

        monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

        response = client.post(
            "/v1/frontend/chat", json={"username": user.username, "question": "What is PaymentCharged?"}
        )

        assert response.status_code == 200
        assert response.json()["diagram"] is None
    finally:
        await _cleanup(user.tenant)
