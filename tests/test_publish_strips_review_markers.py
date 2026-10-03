"""Regression test for a real, user-reported issue: clicking "Publish" used to persist
`agents.graph._DOWNGRADE_MARKER` (` **[unknown — flagged by review]**`) verbatim as part of the
ADR's own permanent record. That marker is an internal review annotation — `boss_decide`
flagging a claim the Critic could not verify against the transcript, shown on the review card
so a human can judge it before publishing — not something a published ADR should carry forever
once a human has accepted it.

`app.routers.frontend.finalize_document` (the endpoint the frontend's "Publish" button calls,
`frontend/app.py`'s `_render_adr_candidates`) now runs `agents.graph.strip_downgrade_markers` on
the incoming content before it touches `SilverDocument`, `SilverChunk`, or Gold's own extraction
LLM call, so every downstream reader sees the same clean text a human accepted, never the raw
review markup."""

from datetime import date
from uuid import uuid4

import litellm
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete, select

from agents.graph import strip_downgrade_markers
from agents.stages.gold.schemas import GoldExtractionResult
from app.config import settings
from app.main import app
from db.models import BronzeDocument, GoldAlias, GoldEvolution, SilverChunk, SilverDocument
from db.session import async_session_factory

pytestmark = pytest.mark.anyio
client = TestClient(app)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _fake_response(content: str):
    from types import SimpleNamespace

    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice], model="fake")


async def _fake_acompletion(*, model, api_key, messages, **kwargs):
    result = GoldExtractionResult(
        components=[], contracts=[], architecture_change="unchanged",
        architecture_narrative="No architecture change extracted for this test.", mermaid_diagram="",
    )
    return _fake_response(result.model_dump_json())


async def _cleanup(tenant: str, source_component: str) -> None:
    async with async_session_factory() as session:
        await session.execute(
            delete(BronzeDocument).where(
                BronzeDocument.tenant == tenant, BronzeDocument.source_component == source_component
            )
        )
        await session.execute(
            delete(SilverDocument).where(
                SilverDocument.tenant == tenant, SilverDocument.source_component == source_component
            )
        )
        await session.execute(
            delete(SilverChunk).where(
                SilverChunk.tenant == tenant, SilverChunk.source_component == source_component
            )
        )
        await session.execute(
            delete(GoldEvolution).where(
                GoldEvolution.tenant == tenant, GoldEvolution.source_component == source_component
            )
        )
        await session.execute(
            delete(GoldAlias).where(
                GoldAlias.tenant == tenant, GoldAlias.source_component == source_component
            )
        )
        await session.commit()


def test_strip_downgrade_markers_restores_the_original_claim_text():
    content = (
        "# ADR\n\nIntroduce a new frontend component **[unknown — flagged by review]**. "
        "It calls the backend **[unknown — flagged by review]**."
    )
    stripped = strip_downgrade_markers(content)
    assert "flagged by review" not in stripped
    assert stripped == "# ADR\n\nIntroduce a new frontend component. It calls the backend."


def test_strip_downgrade_markers_is_a_noop_on_content_with_no_marker():
    content = "# ADR\n\nNothing here was ever flagged."
    assert strip_downgrade_markers(content) == content


async def test_publish_endpoint_strips_the_marker_before_persisting(monkeypatch):
    tenant = settings.frontend_users[0].tenant
    username = settings.frontend_users[0].username
    source_component = f"publish-strip-{uuid4().hex[:8]}.en.vtt"
    ingestion_date = date(2026, 6, 15)

    async with async_session_factory() as session:
        session.add(
            BronzeDocument(
                tenant=tenant,
                ingestion_date=ingestion_date,
                source_component=source_component,
                content="The team introduced a new frontend component.",
                embedding=[0.0] * settings.embedding_dim,
            )
        )
        await session.commit()

    monkeypatch.setattr(litellm, "acompletion", _fake_acompletion)

    try:
        content_with_marker = (
            "# ADR\n\nIntroduce a new frontend component **[unknown — flagged by review]**."
        )
        response = client.post(
            "/v1/frontend/transcriptions/finalize",
            json={"username": username, "source_component": source_component, "content": content_with_marker},
        )
        assert response.status_code == 200

        async with async_session_factory() as session:
            doc = (
                await session.execute(
                    select(SilverDocument).where(
                        SilverDocument.tenant == tenant, SilverDocument.source_component == source_component
                    )
                )
            ).scalars().one()
            chunk = (
                await session.execute(
                    select(SilverChunk).where(
                        SilverChunk.tenant == tenant, SilverChunk.source_component == source_component
                    )
                )
            ).scalars().one()

        assert "flagged by review" not in doc.content
        assert doc.content == "# ADR\n\nIntroduce a new frontend component."
        assert "flagged by review" not in chunk.content
    finally:
        await _cleanup(tenant, source_component)
