"""Covers `.tmp/tasks2.md` task 3: Query Expansion. `expand_question`
(`agents/stages/gold/retrieval/query_expansion.py`) gets its own fast, mocked-LLM unit test;
`top_k_gold_evolution`'s `expand=True` wiring gets a test proving the exact gap this technique
targets — a genuine synonym with ZERO words in common with the canonical name, which neither
plain vector nor plain lexical search can find, but a reformulation that lands on/near the
canonical wording can.

The vector channel is neutralized with `max_distance=0.0` in the wiring test — no two different
texts embed identically, so this reliably excludes every candidate from the vector ranking,
isolating the test to the deterministic lexical channel instead of depending on how close the
real local embedding model happens to consider two different phrasings (which would make the
test flaky either way it failed or passed)."""

import json
from datetime import date
from types import SimpleNamespace
from uuid import uuid4

import litellm
import pytest

from agents.stages.gold import service as gold_service
from agents.stages.gold.service import embed_question, persist_entity_version, top_k_gold_evolution
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


# --- expand_question, in isolation: mocked LLM, no DB --------------------------------------


async def test_expand_question_returns_the_llms_reformulations(monkeypatch):
    captured = {}

    async def fake_acompletion(*, model, api_key, messages, **kwargs):
        captured["messages"] = messages
        captured["kwargs"] = kwargs
        payload = {"reformulations": ["What does the Payments Gateway do?", "Explain the payments gateway service."]}
        return _fake_response(json.dumps(payload))

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    result = await gold_service.expand_question("What is the billing module?")

    assert result == ["What does the Payments Gateway do?", "Explain the payments gateway service."]
    assert "billing module" in captured["messages"][0]["content"]
    # Low but non-zero — see expand_question's own docstring for why temperature=0 is the
    # wrong choice here, unlike every temperature=0 call elsewhere in this codebase.
    assert 0 < captured["kwargs"]["temperature"] < 1


async def test_expand_question_returns_empty_list_when_the_llm_finds_nothing_worth_adding(monkeypatch):
    async def fake_acompletion(*, model, api_key, messages, **kwargs):
        return _fake_response(json.dumps({"reformulations": []}))

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    assert await gold_service.expand_question("What is X?") == []


# --- top_k_gold_evolution(expand=True): the actual gap this technique targets ---------------


async def test_synonym_with_no_shared_words_is_found_only_with_expansion_on(monkeypatch):
    tenant = f"query-expansion-{uuid4().hex[:8]}"
    entity_id = str(uuid4())
    # Deliberately zero lexical overlap with "Payments Gateway" / "outbound charges" — this is
    # the exact case hybrid search alone cannot rescue (see the module docstring).
    original_question = "Tell me about the billing module"
    reformulation = "What does the Payments Gateway do?"

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

        vector = await embed_question(original_question)

        async with async_session_factory() as session:
            without_expansion = await top_k_gold_evolution(
                session,
                vector,
                k=5,
                tenant=tenant,
                mode="hybrid",
                question_text=original_question,
                max_distance=0.0,  # neutralizes the vector channel — see module docstring
                expand=False,
            )
        assert without_expansion == []  # the exact gap this technique exists to close

        async def fake_expand_question(question: str) -> list[str]:
            assert question == original_question
            return [reformulation]

        monkeypatch.setattr(gold_service, "expand_question", fake_expand_question)

        async with async_session_factory() as session:
            with_expansion = await top_k_gold_evolution(
                session,
                vector,
                k=5,
                tenant=tenant,
                mode="hybrid",
                question_text=original_question,
                max_distance=0.0,
                expand=True,
            )
        assert len(with_expansion) == 1
        assert with_expansion[0].entity_id == entity_id
    finally:
        await _cleanup(tenant)
