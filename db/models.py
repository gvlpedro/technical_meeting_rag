from datetime import date, datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import Date, DateTime, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.config import settings


class Base(DeclarativeBase):
    pass


class BronzeDocument(Base):
    """Raw ingested transcript chunks (Bronze layer of the medallion pipeline).

    Silver (clarified, summary-chunked) and Gold (structure-aware, component-graph)
    layers are future work — see README's medallion diagram — not implemented yet.
    """

    __tablename__ = "bronze_documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    # Date the source video was uploaded on YouTube
    ingestion_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    # Filename of the .vtt transcript this chunk was cut from (traceability).
    source_component: Mapped[str] = mapped_column(String, nullable=False)
    # The chunk's actual text, as fed to the embedding model.
    content: Mapped[str] = mapped_column(Text, nullable=False)
    # Vector embedding of `content` (dimension fixed by settings.embedding_dim / the
    # configured sentence-transformers model), used for similarity search.
    embedding: Mapped[list[float]] = mapped_column(Vector(settings.embedding_dim), nullable=False)
    # Row insertion timestamp (when this chunk was ingested, not when it was recorded).
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class SilverDocument(Base):
    """One clarified ADR document per (source transcript, version) — Silver layer.

    Shaped as `prompts/adr_generator.jinja` produces it, filled in as
    far as the clarification loop's answers allow — see `doc/silver_process.md` §1-2.
    `content` is the whole document, not chunked; `SilverChunk` below holds the single
    retrievable chunk cut from it. Silver stores nothing on disk at all — the
    clarification audit trail lives in `SilverClarification` below, not as a column
    here.

    Versioned, not just upserted: `write_document` hashes the freshly generated
    content and compares it against the latest existing row for this
    `source_component`. Identical hash → that same row is overwritten in place (same
    `version`, idempotent re-run). Different hash → a new row is inserted at
    `version + 1`, and the older version's row is left untouched — both stay queryable.
    `(source_component, version)` is the real identity; `source_component` alone is no
    longer unique.

    `mentioned_component_names`/`mentioned_data_contract_names` persist what `generate_
    architecture_questions` identifies for this row's own source — previously computed and
    discarded every run; see the column comments below and `agents.service.
    mentions_grounded_in_source`.
    """

    __tablename__ = "silver_documents"
    __table_args__ = (UniqueConstraint("source_component", "version", name="uq_silver_documents_source_version"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ingestion_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    # Matches bronze_documents.source_component. No longer unique alone — see class
    # docstring; (source_component, version) is the real identity now.
    source_component: Mapped[str] = mapped_column(String, nullable=False, index=True)
    # 1, 2, 3... per source_component — bumped only when the generated content's hash
    # actually changes from the latest stored version (write_document decides this,
    # not the LLM).
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    # sha256 hex digest of `content` — write_document's cheap way to tell "identical
    # re-run" (overwrite this version) from "genuinely new content" (new version)
    # without diffing full document text on every run.
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    # Raw, unresolved component/data-contract mentions grounded in THIS row's own source —
    # `MentionedComponent`/`MentionedDataContract`-shaped dicts (agents/schemas.py), filtered
    # from generate_architecture_questions' batch-wide output down to just the ones whose name
    # appears verbatim in this source_component's transcript content (see
    # agents.service.mentions_grounded_in_source). Named "mentioned", not "canonical" —
    # deliberately not yet resolved through any identity/alias matching; that resolution is
    # Gold-level work this column only feeds, never does itself.
    mentioned_component_names: Mapped[list[dict]] = mapped_column(JSONB, nullable=False, default=list)
    mentioned_data_contract_names: Mapped[list[dict]] = mapped_column(JSONB, nullable=False, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class SilverClarification(Base):
    """History of every clarification question drafted and its answer, per source
    transcript (Silver layer) — `doc/silver_process.md` §2. Append-only, unlike
    `SilverDocument`'s upsert-in-place: `write_document` inserts a fresh row per
    (source_component, question) pair on every run instead of overwriting a prior one,
    so the same question answered differently across two runs keeps both answers on
    record, ordered by `answered_at`. `answer` is `NULL` for a question that was never
    actually answered (`unknown`/`needs_clarification` at the time this row was written).
    """

    __tablename__ = "silver_clarifications"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ingestion_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    # Matches bronze_documents.source_component / silver_documents.source_component.
    source_component: Mapped[str] = mapped_column(String, nullable=False, index=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    answered_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class SilverChunk(Base):
    """The single embedded chunk for one SilverDocument (source_component, version).

    Unlike the rest of this pipeline's chunker-based splitting, an ADR is never split
    into multiple retrieval chunks — one ADR is one chunk, one embedding, so a search
    hit always returns the whole decision record rather than a fragment of it. No FK
    to silver_documents — same flat source_component pattern bronze_documents already
    uses (doc/silver_process.md §2) — but `(source_component, version)` is unique, so
    `chunk_and_embed` can upsert exactly the version it just wrote without touching any
    other version's row.
    """

    __tablename__ = "silver_chunks"
    __table_args__ = (UniqueConstraint("source_component", "version", name="uq_silver_chunks_source_version"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ingestion_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    source_component: Mapped[str] = mapped_column(String, nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(settings.embedding_dim), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
