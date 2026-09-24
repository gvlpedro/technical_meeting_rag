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
    parse_vtt_text,
)
from ingestion.embedder import embed


class NoTranscriptsFoundError(Exception):
    """Raised when `ingest_uploaded_files` is called with no files at all."""


@dataclass
class IngestResult:
    ingestion_date: date
    files_ingested: list[str]
    chunks_created: int


async def ingest_uploaded_files(
    files: list[tuple[str, bytes]], ingestion_date: str, tenant: str, db: AsyncSession, *, uploaded_by: str = ""
) -> IngestResult:
    """Ingest files uploaded straight from the frontend's "Input transcription" tab. This
    never touches disk — `files` is `[(filename, raw_bytes), ...]`. A `.vtt` file goes
    through `parse_vtt_text`. A `.pdf`
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
