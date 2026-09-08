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


async def test_insert_silver_document_and_matching_chunk():
    async with async_session_factory() as session:
        session.add(
            SilverDocument(
                ingestion_date=INGESTION_DATE,
                source_component=SOURCE_COMPONENT,
                content="# Architecture Description / Evolution\n\n...",
            )
        )
        session.add(
            SilverChunk(
                ingestion_date=INGESTION_DATE,
                source_component=SOURCE_COMPONENT,
                content="a chunk of the document above",
                embedding=[0.0] * settings.embedding_dim,
            )
        )
        await session.commit()


async def test_duplicate_source_component_violates_unique_constraint():
    async with async_session_factory() as session:
        session.add(
            SilverDocument(
                ingestion_date=INGESTION_DATE,
                source_component=SOURCE_COMPONENT,
                content="first version",
            )
        )
        await session.commit()

    async with async_session_factory() as session:
        session.add(
            SilverDocument(
                ingestion_date=INGESTION_DATE,
                source_component=SOURCE_COMPONENT,
                content="second version, same source_component",
            )
        )
        with pytest.raises(IntegrityError):
            await session.commit()
