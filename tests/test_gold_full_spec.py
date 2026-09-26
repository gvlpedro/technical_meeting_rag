"""Regression test for a real, user-reported bug: asking for a data contract's full
specification always got "the retrieved facts don't provide it", even when `odcs_spec` was
fully populated — `_payload_detail`'s data_contract branch only ever showed producer/consumer
and schema FIELD NAMES, never the actual spec. `wants_full_spec`/`_full_odcs_spec_block`
(`agents/stages/gold/service.py`) close this gap by conditionally including the entire spec in
`build_context_lines`'s output, only when the question actually asks for it."""

import json
from datetime import date
from types import SimpleNamespace
from uuid import uuid4

import litellm
import pytest
from sqlalchemy import select

from agents.stages.gold.schemas import DataContractPayload
from agents.stages.gold.service import answer_question, build_context_lines, latest_versions, persist_entity_version
from db.models import GoldAlias, GoldEvolution
from db.session import async_session_factory

pytestmark = pytest.mark.anyio


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


async def _seed_contract(tenant: str, contract_id: str, spec: dict) -> None:
    async with async_session_factory() as session:
        await persist_entity_version(
            session, entity_type="data_contract", entity_id=contract_id,
            canonical_name="registration-login-purchase", operation="new",
            narrative="registration-login-purchase carries user registration data.",
            payload=DataContractPayload(producer="frontend", consumer="backend", odcs_spec=spec).model_dump(),
            source_component="adr1.txt", source_adr_version=1, ingestion_date=date(2026, 1, 1), tenant=tenant,
        )
        await session.commit()


async def test_build_context_lines_includes_the_full_spec_only_when_the_question_asks_for_it():
    tenant = f"full-spec-{uuid4().hex[:8]}"
    contract_id = str(uuid4())
    spec = {
        "apiVersion": "odcs/v3.0.0",
        "status": "active",
        "schema": {"properties": {"email": {"description": "User email address."}}},
    }
    try:
        await _seed_contract(tenant, contract_id, spec)

        async with async_session_factory() as session:
            rows = (
                await session.execute(select(GoldEvolution).where(GoldEvolution.entity_id == contract_id))
            ).scalars().all()
            without_ask = await build_context_lines(session, rows, question="What is this contract for?")
            with_ask = await build_context_lines(
                session, rows, question="Could you show me the full specification for this contract?"
            )

        assert "Full specification:" not in without_ask[0]
        assert "Full specification:" in with_ask[0]
        assert '"apiVersion": "odcs/v3.0.0"' in with_ask[0]
    finally:
        await _cleanup(tenant)


async def test_answer_question_gives_the_llm_the_full_spec_when_asked(monkeypatch):
    tenant = f"full-spec-{uuid4().hex[:8]}"
    contract_id = str(uuid4())
    spec = {"apiVersion": "odcs/v3.0.0", "status": "active"}
    try:
        await _seed_contract(tenant, contract_id, spec)

        captured = {}

        async def fake_acompletion(*, model, api_key, messages, **kwargs):
            captured["content"] = messages[0]["content"]
            return _fake_response(json.dumps({"answer": "Here is the full spec.", "citations": []}))

        monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

        async with async_session_factory() as session:
            rows = (
                await session.execute(select(GoldEvolution).where(GoldEvolution.entity_id == contract_id))
            ).scalars().all()
            latest = await latest_versions(session, rows)
            await answer_question(
                session, "Could you show me the full specification for registration-login-purchase?", rows, latest
            )

        assert '"apiVersion": "odcs/v3.0.0"' in captured["content"]
    finally:
        await _cleanup(tenant)
