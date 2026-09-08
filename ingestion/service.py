import asyncio
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from db.models import BronzeDocument
from ingestion.bronze_documents_chunker import chunk_text, parse_ingestion_date, parse_vtt
from ingestion.embedder import embed


class NoTranscriptsFoundError(Exception):
    """No `.vtt` files exist for the requested `ingestion_date` partition."""


@dataclass
class IngestResult:
    ingestion_date: date
    files_ingested: list[str]
    chunks_created: int


async def ingest_bronze(ingestion_date: str, db: AsyncSession) -> IngestResult:
    """Ingest every transcript uploaded on `ingestion_date` (compact `YYYYMMDD`, e.g.
    `20260906`), looked up at `input/transcriptions/ingestion_date=20260906/`.

    Shared by `app/routers/ingestion.py` (`POST /v1/ingest`) and `scripts/ingest.py`
    (`make ingestion`) — one implementation, two entry points. Raises `ValueError` for
    a malformed date, `NoTranscriptsFoundError` when nothing to ingest exists yet.
    """
    parsed_date = parse_ingestion_date(ingestion_date)  # raises ValueError if malformed

    partition = f"ingestion_date={ingestion_date}"
    transcripts_dir = Path(settings.input_dir) / "transcriptions" / partition
    if not transcripts_dir.is_dir():
        raise NoTranscriptsFoundError(f"No transcripts found for '{partition}'")

    vtt_files = sorted(transcripts_dir.glob("*.vtt"))
    if not vtt_files:
        raise NoTranscriptsFoundError(f"No .vtt files found in {transcripts_dir}")

    files_ingested = []
    chunks_created = 0

    for vtt_path in vtt_files:
        text = parse_vtt(vtt_path)
        chunks = chunk_text(text, settings.chunk_size_tokens, settings.chunk_overlap_tokens)
        if not chunks:
            continue

        embeddings = await asyncio.to_thread(embed, chunks)

        for content, embedding in zip(chunks, embeddings, strict=True):
            db.add(
                BronzeDocument(
                    ingestion_date=parsed_date,
                    source_component=vtt_path.name,
                    content=content,
                    embedding=embedding,
                )
            )
        chunks_created += len(chunks)
        files_ingested.append(vtt_path.name)

    await db.commit()

    return IngestResult(
        ingestion_date=parsed_date, files_ingested=files_ingested, chunks_created=chunks_created
    )
