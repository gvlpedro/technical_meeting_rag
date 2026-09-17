import asyncio
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from db.models import BronzeDocument
from ingestion.bronze_documents_chunker import (
    chunk_text,
    extract_pdf_text,
    parse_ingestion_date,
    parse_vtt,
    parse_vtt_text,
)
from ingestion.embedder import embed


class NoTranscriptsFoundError(Exception):
    """No `.vtt` files exist for the requested `ingestion_date` partition."""


@dataclass
class IngestResult:
    ingestion_date: date
    files_ingested: list[str]
    chunks_created: int


async def ingest_bronze(ingestion_date: str, db: AsyncSession, *, tenant: str = "default") -> IngestResult:
    """Ingest every transcript uploaded on `ingestion_date` (compact `YYYYMMDD`, e.g.
    `20260906`), looked up at `input/transcriptions/ingestion_date=20260906/`.

    Shared by `app/routers/ingestion.py` (`POST /v1/ingest`) and `scripts/ingest.py`
    (`make ingestion`) — one implementation, two entry points. Raises `ValueError` for
    a malformed date, `NoTranscriptsFoundError` when nothing to ingest exists yet.

    `tenant` tags every row written here (defaults to `"default"` — this disk-based path
    predates the frontend and has no real per-tenant folder layout; the frontend's own upload
    flow goes through `ingest_uploaded_files` below instead, which always has a real tenant).
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
                    tenant=tenant,
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


async def ingest_uploaded_files(
    files: list[tuple[str, bytes]], ingestion_date: str, tenant: str, db: AsyncSession, *, uploaded_by: str = ""
) -> IngestResult:
    """Ingest files uploaded straight from the frontend's "Input transcription" tab — no disk
    partition involved, unlike `ingest_bronze` above. `files` is `[(filename, raw_bytes), ...]`;
    `.vtt` goes through `parse_vtt_text`, `.pdf` through `extract_pdf_text`, `.txt`/`.md` are
    decoded as plain text directly (no cue-timing/metadata stripping needed), anything else is
    rejected with `ValueError` naming the file.

    `uploaded_by` is the logged-in username that submitted this batch — stamped on every
    `BronzeDocument` row written here (see that column's own docstring in `db/models.py`).

    Raises `ValueError` for a malformed `ingestion_date` or an unsupported file extension,
    `NoTranscriptsFoundError` if `files` is empty."""
    parsed_date = parse_ingestion_date(ingestion_date)
    if not files:
        raise NoTranscriptsFoundError("No files were uploaded")

    files_ingested = []
    chunks_created = 0

    for filename, content in files:
        suffix = Path(filename).suffix.lower()
        if suffix == ".vtt":
            text = parse_vtt_text(content.decode("utf-8"))
        elif suffix == ".pdf":
            text = extract_pdf_text(content)
        elif suffix in (".txt", ".md"):
            # Already plain text — no cue-timing/metadata stripping needed, unlike .vtt.
            text = content.decode("utf-8")
        else:
            raise ValueError(
                f"Unsupported file type for {filename!r} — only .txt, .vtt, .md, and .pdf are accepted"
            )

        chunks = chunk_text(text, settings.chunk_size_tokens, settings.chunk_overlap_tokens)
        if not chunks:
            continue

        embeddings = await asyncio.to_thread(embed, chunks)
        for chunk_content, embedding in zip(chunks, embeddings, strict=True):
            db.add(
                BronzeDocument(
                    tenant=tenant,
                    ingestion_date=parsed_date,
                    source_component=filename,
                    uploaded_by=uploaded_by,
                    content=chunk_content,
                    embedding=embedding,
                )
            )
        chunks_created += len(chunks)
        files_ingested.append(filename)

    await db.commit()

    return IngestResult(
        ingestion_date=parsed_date, files_ingested=files_ingested, chunks_created=chunks_created
    )
