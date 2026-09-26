"""Covers `.tmp/tasks2.md` task 2: Corrective RAG. `retrieve_with_correction`
(`agents/stages/gold/retrieval/corrective_rag.py`) gets its own fast, no-DB unit tests; the
`/chat` endpoint's wiring of it (`app/routers/frontend.py`) gets an end-to-end test proving the
retry actually fires with a relaxed `max_distance`, and that a genuinely empty case still
returns the honest "no relevant facts" message instead of a false positive."""

import json
from datetime import date
from types import SimpleNamespace
from uuid import uuid4

import litellm
import pytest
from fastapi.testclient import TestClient

from agents.stages import gold
from agents.stages.gold.retrieval.corrective_rag import retrieve_with_correction
from agents.stages.gold.service import DEFAULT_MAX_DISTANCE, persist_entity_version
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


# --- retrieve_with_correction, in isolation: no DB, no LLM, no FastAPI ----------------------


async def test_retrieve_with_correction_never_retries_when_the_first_attempt_finds_rows():
    calls = []

    async def retrieve(max_distance):
        calls.append(max_distance)
        return ["row"]

    result = await retrieve_with_correction(retrieve, max_distance=0.6, relaxed_max_distance=None)

    assert result == ["row"]
    assert calls == [0.6]  # relaxed attempt never even evaluated


async def test_retrieve_with_correction_retries_once_with_the_relaxed_bound():
    calls = []

    async def retrieve(max_distance):
        calls.append(max_distance)
        return [] if max_distance == 0.6 else ["found on the relaxed retry"]

    result = await retrieve_with_correction(retrieve, max_distance=0.6, relaxed_max_distance=None)

    assert result == ["found on the relaxed retry"]
    assert calls == [0.6, None]


async def test_retrieve_with_correction_returns_empty_when_both_attempts_find_nothing():
    calls = []

    async def retrieve(max_distance):
        calls.append(max_distance)
        return []

    result = await retrieve_with_correction(retrieve, max_distance=0.6, relaxed_max_distance=None)

    assert result == []
    assert calls == [0.6, None]  # still only one retry, not a loop


# --- /chat endpoint wiring: real request, mocked retrieval + mocked LLM ---------------------


async def test_chat_retries_with_a_relaxed_max_distance_before_giving_up(monkeypatch):
    user = settings.frontend_users[0]
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
                tenant=user.tenant,
            )
            await session.commit()

        seen_max_distances = []
        real_top_k = gold.top_k_gold_evolution

        async def tracking_top_k(*args, max_distance=None, **kwargs):
            seen_max_distances.append(max_distance)
            if max_distance == DEFAULT_MAX_DISTANCE:
                return []  # the strict first pass finds nothing
            # The relaxed retry: delegate to the real function, unfiltered, scoped to our row.
            return await real_top_k(*args, max_distance=max_distance, **kwargs)

        monkeypatch.setattr(gold, "top_k_gold_evolution", tracking_top_k)

        async def fake_acompletion(*, model, api_key, messages, **kwargs):
            payload = {"answer": "Payments Gateway handles outbound charges.", "citations": []}
            return _fake_response(json.dumps(payload))

        monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

        response = client.post(
            "/v1/frontend/chat", json={"username": user.username, "question": "What is Payments Gateway?"}
        )

        assert response.status_code == 200
        assert seen_max_distances == [DEFAULT_MAX_DISTANCE, None]
        assert response.json()["answer"] == "Payments Gateway handles outbound charges."
    finally:
        await _cleanup(user.tenant)


async def test_chat_gives_the_honest_no_facts_answer_when_both_attempts_are_empty(monkeypatch):
    user = settings.frontend_users[0]
    seen_max_distances = []

    async def empty_top_k(*args, max_distance=None, **kwargs):
        seen_max_distances.append(max_distance)
        return []

    monkeypatch.setattr(gold, "top_k_gold_evolution", empty_top_k)

    response = client.post(
        "/v1/frontend/chat",
        json={"username": user.username, "question": f"anything about {uuid4().hex}?"},
    )

    assert response.status_code == 200
    assert seen_max_distances == [DEFAULT_MAX_DISTANCE, None]  # both attempts really ran
    assert response.json()["answer"] == "No Gold facts were relevant to this question."
