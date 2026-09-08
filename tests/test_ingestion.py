from datetime import date
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete, func, select

from app.config import settings
from app.main import app
from db.models import BronzeDocument
from db.session import async_session_factory
from ingestion.bronze_documents_chunker import chunk_text, parse_vtt

client = TestClient(app)

INGESTION_DATE_VALUE = "20260515"
INGESTION_DATE = date(2026, 5, 15)
PARTITION = f"ingestion_date={INGESTION_DATE_VALUE}"
SOURCE_FILE = "this_is_how_twitter_works_internally.en.vtt"

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


async def test_ingest_creates_expected_chunk_count_with_embeddings():
    vtt_path = Path(settings.input_dir) / "transcriptions" / PARTITION / SOURCE_FILE
    expected_chunks = chunk_text(
        parse_vtt(vtt_path), settings.chunk_size_tokens, settings.chunk_overlap_tokens
    )

    # INGESTION_DATE is a real sample date, also handy for manual `make ingestion` /
    # `make clarify` runs — bronze_documents has no unique constraint, so a plain
    # "delete this date, ingest, assert, delete this date again" fixture would erase
    # whatever a manual run had already ingested there. Watermark by id instead: only
    # count/clean up the rows *this* call creates, leaving any pre-existing ones alone.
    async with async_session_factory() as session:
        watermark = await session.scalar(select(func.max(BronzeDocument.id))) or 0

    try:
        response = client.post("/v1/ingest", params={"ingestion_date": INGESTION_DATE_VALUE})

        assert response.status_code == 200
        assert response.json()["ingestion_date"] == INGESTION_DATE.isoformat()
        assert SOURCE_FILE in response.json()["files_ingested"]

        async with async_session_factory() as session:
            result = await session.execute(
                select(BronzeDocument).where(
                    BronzeDocument.id > watermark,
                    BronzeDocument.ingestion_date == INGESTION_DATE,
                    BronzeDocument.source_component == SOURCE_FILE,
                )
            )
            rows = result.scalars().all()

        assert len(rows) == len(expected_chunks)
        assert len(rows) > 0
        assert all(row.embedding is not None for row in rows)
        assert all(len(row.embedding) == settings.embedding_dim for row in rows)
    finally:
        async with async_session_factory() as session:
            await session.execute(delete(BronzeDocument).where(BronzeDocument.id > watermark))
            await session.commit()


async def test_ingest_unknown_ingestion_date_returns_404():
    response = client.post("/v1/ingest", params={"ingestion_date": "99990101"})

    assert response.status_code == 404


async def test_ingest_malformed_ingestion_date_returns_400():
    response = client.post("/v1/ingest", params={"ingestion_date": "not-a-date"})

    assert response.status_code == 400
