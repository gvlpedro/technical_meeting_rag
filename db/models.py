from datetime import date, datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import Date, DateTime, Integer, String, Text, func
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
    """One clarified document per source transcript (Silver layer).

    Shaped as `doc/clarification_template.md` itself, filled in as far as the
    clarification loop's answers allow — see `doc/silver_process.md` §1-2. `content`
    is the whole document, not chunked; `SilverChunk` below holds the retrievable
    pieces cut from it. Silver stores nothing on disk at all — the clarification
    audit trail lives in `SilverClarification` below, not as a column here.
    """

    __tablename__ = "silver_documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ingestion_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    # Matches bronze_documents.source_component. Unique so a re-run upserts this row
    # in place instead of duplicating it (idempotency, doc/silver_process.md §1).
    source_component: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
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
    """Chunked + embedded pieces of a SilverDocument's content, for retrieval.

    No FK to silver_documents — same flat source_component pattern bronze_documents
    already uses (doc/silver_process.md §2).
    """

    __tablename__ = "silver_chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ingestion_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    source_component: Mapped[str] = mapped_column(String, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(settings.embedding_dim), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
