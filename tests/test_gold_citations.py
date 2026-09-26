"""Covers `.tmp/tasks2.md` task 1: `answer_question`/`answer_evolution_question` return a
structured `GroundedAnswer` (answer + citations), and every citation is checked against the
rows actually retrieved before it reaches the caller — see `_verify_citations` and
`agents.stages.gold.schemas.GroundedAnswer`. These are fast, mocked-LLM tests (`make test`),
not a real-LLM golden-set case."""

import json
from datetime import date
from types import SimpleNamespace
from uuid import uuid4

import litellm
import pytest
from sqlalchemy import select

from agents.stages.gold.schemas import ChatCitation
from agents.stages.gold.service import _verify_citations, answer_question, persist_entity_version
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


# --- _verify_citations, in isolation: no DB, no LLM -----------------------------------------


def test_verify_citations_keeps_only_citations_matching_a_retrieved_row():
    real_row = GoldEvolution(
        tenant="t", entity_type="component", entity_id="real-id", canonical_name="Real",
        version=2, operation="modified", narrative="n", payload={}, embedding=[0.0],
        entity_hash="h", source_component="s", source_adr_version=1, ingestion_date=date(2026, 1, 1),
    )
    valid = ChatCitation(entity_type="component", entity_id="real-id", version=2)
    fabricated = ChatCitation(entity_type="component", entity_id="made-up-id", version=1)
    wrong_version = ChatCitation(entity_type="component", entity_id="real-id", version=99)

    kept = _verify_citations([valid, fabricated, wrong_version], [real_row])

    assert kept == [valid]


def test_verify_citations_returns_empty_for_no_rows():
    fabricated = ChatCitation(entity_type="component", entity_id="made-up-id", version=1)
    assert _verify_citations([fabricated], []) == []


# --- answer_question, end to end: real row, mocked LLM --------------------------------------


async def test_answer_question_strips_a_fabricated_citation(monkeypatch):
    """The model is made to cite one real row and one row that was never retrieved. Only the
    real one may survive into the returned `GroundedAnswer.citations`."""
    tenant = f"citations-test-{uuid4().hex[:8]}"
    entity_id = str(uuid4())
    try:
        async with async_session_factory() as session:
            await persist_entity_version(
                session,
                entity_type="component",
                entity_id=entity_id,
                canonical_name="Payments Gateway",
                operation="new",
                narrative="Payments Gateway handles outbound charges.",
                payload={"dependency_ids": [], "contract_ids": []},
                source_component="meeting.en.vtt",
                source_adr_version=1,
                ingestion_date=date(2026, 6, 1),
                tenant=tenant,
            )
            await session.commit()

        async def fake_acompletion(*, model, api_key, messages, **kwargs):
            payload = {
                "answer": "Payments Gateway is a new component that handles outbound charges.",
                "citations": [
                    {"entity_type": "component", "entity_id": entity_id, "version": 1},
                    {"entity_type": "component", "entity_id": "never-retrieved", "version": 1},
                ],
            }
            return _fake_response(json.dumps(payload))

        monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

        async with async_session_factory() as session:
            rows = list(
                (
                    await session.execute(
                        select(GoldEvolution).where(
                            GoldEvolution.tenant == tenant, GoldEvolution.entity_id == entity_id
                        )
                    )
                )
                .scalars()
                .all()
            )
            assert len(rows) == 1  # sanity: exactly the one row we just persisted

            result = await answer_question(session, "What is Payments Gateway?", rows)

        assert result.answer == "Payments Gateway is a new component that handles outbound charges."
        assert len(result.citations) == 1
        assert result.citations[0].entity_id == entity_id
    finally:
        await _cleanup(tenant)


async def test_answer_question_with_no_rows_returns_empty_citations():
    # `answer_question` returns before ever touching `session` when `rows` is empty — see its
    # own early-return — so a real session is not needed here.
    result = await answer_question(None, "anything?", [])  # type: ignore[arg-type]
    assert result.answer == "No relevant Gold facts were found for this question."
    assert result.citations == []
