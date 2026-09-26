"""Regression test for a real, user-reported bug: asking about a component's facts "in version
N" got answered from the LATEST version instead, because `top_k_gold_evolution` only ever
returns the latest version per entity (by design, for "current state" questions) — a question
pinned to an explicit past version had no path to reach it, even though `entity_history` already
had the data. `app/routers/frontend.py::_answer_specific_version_question` closes this gap."""

import json
from datetime import date
from types import SimpleNamespace
from uuid import uuid4

import litellm
import pytest
from fastapi.testclient import TestClient

from agents.stages.gold.schemas import ComponentPayload
from agents.stages.gold.service import ensure_alias, persist_entity_version
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


async def _seed_backend_with_two_versions(tenant: str) -> tuple[str, str, str]:
    """v1: input contract is `registration-only`. v4 (latest): input contract is
    `registration-login-purchase` — matches the real reported bug's own data shape, where the
    latest version's input contract differs from an earlier one's."""
    backend_id, v1_contract_id, v4_contract_id = str(uuid4()), str(uuid4()), str(uuid4())
    async with async_session_factory() as session:
        await persist_entity_version(
            session, entity_type="component", entity_id=backend_id, canonical_name="backend",
            operation="new", narrative="backend is newly introduced.",
            payload=ComponentPayload(input_contract_ids=[v1_contract_id]).model_dump(),
            source_component="adr1.txt", source_adr_version=1, ingestion_date=date(2026, 1, 1), tenant=tenant,
        )
        for version, source in ((2, "adr2.txt"), (3, "adr3.txt")):
            await persist_entity_version(
                session, entity_type="component", entity_id=backend_id, canonical_name="backend",
                operation="modified", narrative=f"backend was modified again, step {version}.",
                payload=ComponentPayload(input_contract_ids=[v1_contract_id]).model_dump(),
                source_component=source, source_adr_version=1, ingestion_date=date(2026, 1, version), tenant=tenant,
            )
        await persist_entity_version(
            session, entity_type="component", entity_id=backend_id, canonical_name="backend",
            operation="modified", narrative="backend now uses a different input contract.",
            payload=ComponentPayload(input_contract_ids=[v4_contract_id]).model_dump(),
            source_component="adr4.txt", source_adr_version=1, ingestion_date=date(2026, 1, 4), tenant=tenant,
        )
        await ensure_alias(session, "component", backend_id, "backend", "adr1.txt", 1, tenant=tenant)
        await session.commit()
    return backend_id, v1_contract_id, v4_contract_id


async def test_specific_version_question_answers_from_the_pinned_version_not_the_latest(monkeypatch):
    tenant = settings.frontend_users[0].tenant
    try:
        await _seed_backend_with_two_versions(tenant)

        async def fake_acompletion(*, model, api_key, messages, **kwargs):
            content_in = messages[0]["content"]
            # A cheap stand-in for the real LLM: echoes back whichever version's content it was
            # actually given, so the assertion below can tell v1's context from v4's.
            answer = "backend's version 1 input contract is registration-only." if "(version 1," in content_in else (
                "backend's input contract is something else."
            )
            return _fake_response(json.dumps({"answer": answer, "citations": []}))

        monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

        response = client.post(
            "/v1/frontend/chat",
            json={
                "username": settings.frontend_users[0].username,
                "question": "could you list me all input contracts for backend in version 1?",
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert body["answer"] == "backend's version 1 input contract is registration-only."
        assert body["retrieved"] == [
            {"entity_type": "component", "canonical_name": "backend", "version": 1, "operation": "new"}
        ]
    finally:
        await _cleanup(tenant)


async def test_specific_version_question_is_honest_when_that_version_does_not_exist():
    tenant = settings.frontend_users[0].tenant
    try:
        await _seed_backend_with_two_versions(tenant)

        response = client.post(
            "/v1/frontend/chat",
            json={"username": settings.frontend_users[0].username, "question": "what was backend like in version 9?"},
        )

        assert response.status_code == 200
        body = response.json()
        assert "no version 9" in body["answer"].lower()
        assert "1, 2, 3, 4" in body["answer"]
    finally:
        await _cleanup(tenant)


async def test_a_bare_version_question_without_a_number_is_not_pinned():
    """"What version is backend on?" must NOT be hijacked by this path — no explicit number is
    named, so `top_k_gold_evolution`'s own `[latest version]` tag already answers it fine."""
    async with async_session_factory() as session:
        from app.routers.frontend import ChatRequest, _answer_specific_version_question

        result = await _answer_specific_version_question(
            session, "default", ChatRequest(username="x", question="What version is backend on?"), []
        )
    assert result is None


async def test_a_pinned_version_with_no_matching_entity_is_not_handled_here():
    async with async_session_factory() as session:
        from app.routers.frontend import ChatRequest, _answer_specific_version_question

        result = await _answer_specific_version_question(
            session, "default", ChatRequest(username="x", question=f"tell me about {uuid4().hex} in version 1"), []
        )
    assert result is None
