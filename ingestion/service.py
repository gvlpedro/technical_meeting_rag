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
    """Ingest every transcript uploaded on `ingestion_date`. Use the compact `YYYYMMDD`
    format, for example `20260906`. This looks up files at
    `input/transcriptions/ingestion_date=20260906/`.

    Both `app/routers/ingestion.py` (`POST /v1/ingest`) and `scripts/ingest.py`
    (`make ingestion`) share this one function as their implementation. So there is one
    implementation and two entry points. This raises `ValueError` for a malformed date. It
    raises `NoTranscriptsFoundError` when there is nothing to ingest yet.

    `tenant` tags every row written here. It defaults to `"default"`. This disk-based path
    predates the frontend and has no real per-tenant folder layout. The frontend's own
    upload flow goes through `ingest_uploaded_files` below instead. That flow always has a
    real tenant.
    """
    parsed_date = parse_ingestion_date(ingestion_date)  # raises ValueError if the date is not well formed

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
    """Ingest files uploaded straight from the frontend's "Input transcription" tab. Unlike
    `ingest_bronze` above, this does not use any disk partition. `files` is
    `[(filename, raw_bytes), ...]`. A `.vtt` file goes through `parse_vtt_text`. A `.pdf`
    file goes through `extract_pdf_text`. A `.txt` or `.md` file is decoded as plain text
    directly, since it needs no cue-timing or metadata stripping. Any other file type is
    rejected with a `ValueError` that names the file.

    `uploaded_by` is the logged-in username that submitted this batch. It is stamped on
    every `BronzeDocument` row written here. See that column's own docstring in
    `db/models.py`.

    Raises `ValueError` for a malformed `ingestion_date` or an unsupported file extension.
    Raises `NoTranscriptsFoundError` if `files` is empty."""
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
            # This is already plain text. Unlike .vtt, it needs no cue-timing or metadata stripping.
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
