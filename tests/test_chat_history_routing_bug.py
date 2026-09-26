"""Regression test for a real, user-reported bug: `/chat` routes to the wrong path when a
marker word happens to appear in PRIOR chat history — even the assistant's own earlier answer,
not just the user's words.

`app/routers/frontend.py`'s `chat` endpoint used to run `gold.is_evolution_question` and
`gold.find_entity_by_name_in_text` against `contextualized_question` (the current question with
recent history prefixed onto it, meant only for embedding/lexical retrieval). A word like
"historically" in an earlier turn flipped `is_evolution_question` to `True` for a completely
unrelated follow-up, and `find_entity_by_name_in_text`'s "longest alias wins" rule could then
match some OTHER entity named in that stale history instead of the one the CURRENT question
actually names — sending the user a full-history answer about the wrong entity, or a bogus "no
data for X" answer. Both checks must react only to `request.question`, never
`contextualized_question`."""

import json
from datetime import date
from types import SimpleNamespace
from uuid import uuid4

import litellm
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from agents.stages import gold
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


async def test_a_marker_word_in_prior_chat_history_never_hijacks_the_current_question(monkeypatch):
    user = settings.frontend_users[0]
    checkout_id = str(uuid4())
    try:
        async with async_session_factory() as session:
            await persist_entity_version(
                session,
                entity_type="component",
                entity_id=checkout_id,
                canonical_name="Checkout Service",
                operation="new",
                narrative="Checkout Service handles the checkout flow.",
                payload={"dependency_ids": [], "contract_ids": []},
                source_component="meeting.en.vtt",
                source_adr_version=1,
                ingestion_date=date(2026, 6, 1),
                tenant=user.tenant,
            )
            await session.commit()

        async def fake_top_k(*args, **kwargs):
            async with async_session_factory() as session:
                return list(
                    (
                        await session.execute(select(GoldEvolution).where(GoldEvolution.entity_id == checkout_id))
                    ).scalars()
                )

        monkeypatch.setattr(gold, "top_k_gold_evolution", fake_top_k)

        def explode_if_called(*args, **kwargs):
            raise AssertionError(
                "entity_history must not run: the CURRENT question is not an evolution question, "
                "even though an earlier chat turn's text contains an evolution marker word"
            )

        monkeypatch.setattr(gold, "entity_history", explode_if_called)

        async def fake_acompletion(*, model, api_key, messages, **kwargs):
            payload = {"answer": "Checkout Service handles the checkout flow.", "citations": []}
            return _fake_response(json.dumps(payload))

        monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

        response = client.post(
            "/v1/frontend/chat",
            json={
                "username": user.username,
                "question": "What does the Checkout Service do?",
                # The assistant's OWN earlier answer, not the user's — proving this isn't about
                # sanitizing user input, `contextualized_question` prefixes every role's content.
                "history": [
                    {"role": "assistant", "content": "Historically, the Payments Gateway handled outbound charges."}
                ],
            },
        )

        assert response.status_code == 200
        assert response.json()["answer"] == "Checkout Service handles the checkout flow."
    finally:
        await _cleanup(user.tenant)


async def test_a_longer_alias_in_prior_history_never_outranks_the_entity_this_question_names(monkeypatch):
    """Even for a genuine evolution question, the entity to fetch history for must come from
    THIS question, not from a longer, unrelated alias sitting in earlier chat turns."""
    tenant = settings.frontend_users[0].tenant
    checkout_id, payments_gateway_id = str(uuid4()), str(uuid4())
    try:
        async with async_session_factory() as session:
            await persist_entity_version(
                session, entity_type="component", entity_id=checkout_id, canonical_name="Checkout Service",
                operation="new", narrative="Checkout Service handles the checkout flow.",
                payload={"dependency_ids": [], "contract_ids": []}, source_component="meeting.en.vtt",
                source_adr_version=1, ingestion_date=date(2026, 6, 1), tenant=tenant,
            )
            await persist_entity_version(
                session, entity_type="component", entity_id=payments_gateway_id,
                canonical_name="Payments Gateway Extended Processing Service", operation="new",
                narrative="Handles outbound charges.", payload={"dependency_ids": [], "contract_ids": []},
                source_component="meeting.en.vtt", source_adr_version=1, ingestion_date=date(2026, 6, 1),
                tenant=tenant,
            )
            # `persist_entity_version` alone never writes `gold_aliases` — that only happens
            # through `resolve_and_alias`/`ensure_alias`, one step earlier in the real pipeline.
            # `find_entity_by_name_in_text` reads `gold_aliases`, not `GoldEvolution` directly.
            await ensure_alias(session, "component", checkout_id, "Checkout Service", "meeting.en.vtt", 1, tenant=tenant)
            await ensure_alias(
                session, "component", payments_gateway_id, "Payments Gateway Extended Processing Service",
                "meeting.en.vtt", 1, tenant=tenant,
            )
            await session.commit()

        seen_entity_ids = []
        real_entity_history = gold.entity_history

        async def tracking_entity_history(session, entity_type, entity_id, *, tenant):
            seen_entity_ids.append(entity_id)
            return await real_entity_history(session, entity_type, entity_id, tenant=tenant)

        monkeypatch.setattr(gold, "entity_history", tracking_entity_history)

        async def fake_acompletion(*, model, api_key, messages, **kwargs):
            return _fake_response(json.dumps({"answer": "Checkout Service was introduced once.", "citations": []}))

        monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

        response = client.post(
            "/v1/frontend/chat",
            json={
                "username": settings.frontend_users[0].username,
                "question": "What is the history of the Checkout Service?",
                "history": [
                    {
                        "role": "assistant",
                        "content": "The Payments Gateway Extended Processing Service handles outbound charges.",
                    }
                ],
            },
        )

        assert response.status_code == 200
        assert seen_entity_ids == [checkout_id]  # never the longer alias sitting in prior history
    finally:
        await _cleanup(tenant)
