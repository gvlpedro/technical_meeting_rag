import asyncio
from datetime import date
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from db.models import BronzeDocument
from db.session import get_session
from ingestion.bronze_documents_chunker import chunk_text, parse_ingestion_date, parse_vtt
from ingestion.embedder import embed

router = APIRouter(prefix="/v1", tags=["ingestion"])


class IngestResponse(BaseModel):
    ingestion_date: date
    files_ingested: list[str]
    chunks_created: int


@router.post("/ingest", response_model=IngestResponse)
async def ingest(
    ingestion_date: str, db: AsyncSession = Depends(get_session)
) -> IngestResponse:
    """Ingest every transcript uploaded on `ingestion_date` (compact `YYYYMMDD`, e.g.
    `?ingestion_date=20260906`), looked up at `input/transcriptions/ingestion_date=20260906/`."""
    try:
        parsed_date = parse_ingestion_date(ingestion_date)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid ingestion_date '{ingestion_date}', expected YYYYMMDD",
        )

    partition = f"ingestion_date={ingestion_date}"
    transcripts_dir = Path(settings.input_dir) / "transcriptions" / partition
    if not transcripts_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"No transcripts found for '{partition}'")

    vtt_files = sorted(transcripts_dir.glob("*.vtt"))
    if not vtt_files:
        raise HTTPException(status_code=404, detail=f"No .vtt files found in {transcripts_dir}")

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

    return IngestResponse(
        ingestion_date=parsed_date, files_ingested=files_ingested, chunks_created=chunks_created
    )
