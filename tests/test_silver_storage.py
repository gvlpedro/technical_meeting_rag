import hashlib
from datetime import date

import pytest
from sqlalchemy import delete
from sqlalchemy.exc import IntegrityError

from app.config import settings
from db.models import SilverChunk, SilverDocument
from db.session import async_session_factory

INGESTION_DATE = date(2026, 5, 15)
SOURCE_COMPONENT = "this_is_how_twitter_works_internally.en"

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(autouse=True)
async def cleanup():
    yield
    async with async_session_factory() as session:
        await session.execute(
            delete(SilverChunk).where(SilverChunk.ingestion_date == INGESTION_DATE)
        )
        await session.execute(
            delete(SilverDocument).where(SilverDocument.ingestion_date == INGESTION_DATE)
        )
        await session.commit()


def _hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


async def test_insert_silver_document_and_matching_chunk():
    async with async_session_factory() as session:
        session.add(
            SilverDocument(
                ingestion_date=INGESTION_DATE,
                source_component=SOURCE_COMPONENT,
                content="# ADR — ...",
                content_hash=_hash("# ADR — ..."),
            )
        )
        session.add(
            SilverChunk(
                ingestion_date=INGESTION_DATE,
                source_component=SOURCE_COMPONENT,
                content="the whole ADR, as one chunk",
                embedding=[0.0] * settings.embedding_dim,
            )
        )
        await session.commit()


async def test_same_source_component_and_version_violates_unique_constraint():
    """Two `SilverDocument` rows for the same `source_component` both default to
    `version=1` unless told otherwise — the exact "identical re-run" case
    `write_document`'s hash comparison is meant to catch before it ever reaches the
    database. The (`source_component`, `version`) unique constraint is the last line
    of defense against that, not `source_component` alone anymore."""
    async with async_session_factory() as session:
        session.add(
            SilverDocument(
                ingestion_date=INGESTION_DATE,
                source_component=SOURCE_COMPONENT,
                content="first version",
                content_hash=_hash("first version"),
            )
        )
        await session.commit()

    async with async_session_factory() as session:
        session.add(
            SilverDocument(
                ingestion_date=INGESTION_DATE,
                source_component=SOURCE_COMPONENT,
                content="second version, same source_component, same (default) version number",
                content_hash=_hash("second version, same source_component, same (default) version number"),
            )
        )
        with pytest.raises(IntegrityError):
            await session.commit()


async def test_a_second_document_version_for_the_same_source_component_is_allowed():
    """The whole point of moving off a plain `source_component`-unique constraint:
    two genuinely different ADRs for the same transcript, at different `version`
    numbers, must both persist — this is `write_document`'s "new version" branch,
    exercised here directly against the schema rather than through the graph."""
    async with async_session_factory() as session:
        session.add(
            SilverDocument(
                ingestion_date=INGESTION_DATE,
                source_component=SOURCE_COMPONENT,
                version=1,
                content="v1 content",
                content_hash=_hash("v1 content"),
            )
        )
        session.add(
            SilverDocument(
                ingestion_date=INGESTION_DATE,
                source_component=SOURCE_COMPONENT,
                version=2,
                content="v2 content, genuinely different from v1",
                content_hash=_hash("v2 content, genuinely different from v1"),
            )
        )
        await session.commit()  # no IntegrityError — different version, same source_component
