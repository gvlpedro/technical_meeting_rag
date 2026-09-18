import hashlib
from datetime import date

import pytest
from sqlalchemy import delete, select
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
    `version=1`, unless told otherwise. This is the exact "identical re-run" case.
    `write_document`'s hash comparison should catch this case before it ever reaches
    the database. The (`source_component`, `version`) unique constraint is the last
    line of defense against that. `source_component` alone is no longer the unique
    constraint."""
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


async def test_mentioned_names_default_to_empty_list_and_round_trip():
    """`mentioned_component_names` and `mentioned_data_contract_names` are NOT NULL columns
    with a Python-side default. A row built without passing them must not fail. Every
    pre-existing call site builds rows this way. Such a row must default to `[]`, not
    `None`. A row that does pass real `MentionedComponent`/`MentionedDataContract`-shaped
    dicts must round-trip them exactly through Postgres' `jsonb` column."""
    async with async_session_factory() as session:
        bare = SilverDocument(
            ingestion_date=INGESTION_DATE,
            source_component=SOURCE_COMPONENT,
            content="# ADR — no mentions passed",
            content_hash=_hash("# ADR — no mentions passed"),
        )
        session.add(bare)
        await session.commit()
        await session.refresh(bare)
        assert bare.mentioned_component_names == []
        assert bare.mentioned_data_contract_names == []

    grounded_components = [{"name": "checkout service", "status": "new"}]
    grounded_contracts = [
        {"name": "checkout-completed", "producer": "checkout service", "consumer": "unknown", "action": "new"}
    ]
    async with async_session_factory() as session:
        session.add(
            SilverDocument(
                ingestion_date=INGESTION_DATE,
                source_component=SOURCE_COMPONENT,
                version=2,
                content="# ADR — with mentions",
                content_hash=_hash("# ADR — with mentions"),
                mentioned_component_names=grounded_components,
                mentioned_data_contract_names=grounded_contracts,
            )
        )
        await session.commit()

    async with async_session_factory() as session:
        row = (
            await session.execute(
                select(SilverDocument).where(
                    SilverDocument.source_component == SOURCE_COMPONENT, SilverDocument.version == 2
                )
            )
        ).scalar_one()
        assert row.mentioned_component_names == grounded_components
        assert row.mentioned_data_contract_names == grounded_contracts


async def test_a_second_document_version_for_the_same_source_component_is_allowed():
    """This is the whole point of moving off a plain `source_component`-unique constraint.
    Two genuinely different ADRs for the same transcript, at different `version`
    numbers, must both persist. This is `write_document`'s "new version" branch. This
    test exercises that branch directly against the schema, not through the graph."""
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
        await session.commit()  # No IntegrityError here. Different version, same source_component.
