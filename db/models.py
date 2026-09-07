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
