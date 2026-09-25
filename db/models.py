from datetime import date, datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import Computed, Date, DateTime, Float, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.config import settings


class Base(DeclarativeBase):
    pass


class BronzeDocument(Base):
    """Raw ingested transcript chunks. This is the Bronze layer"""

    __tablename__ = "bronze_documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    tenant: Mapped[str] = mapped_column(String, nullable=False, index=True, server_default="default")
    ingestion_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    source_component: Mapped[str] = mapped_column(String, nullable=False)
    uploaded_by: Mapped[str] = mapped_column(String, nullable=False, server_default="")
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(settings.embedding_dim), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class SilverDocument(Base):
    """One clarified ADR document per source transcript"""

    __tablename__ = "silver_documents"
    __table_args__ = (
        UniqueConstraint("tenant", "source_component", "version", name="uq_silver_documents_source_version"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    tenant: Mapped[str] = mapped_column(String, nullable=False, index=True, server_default="default")
    ingestion_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    source_component: Mapped[str] = mapped_column(String, nullable=False, index=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    # This is the sha256 hex to compare docs and overwrite reruns (ssame versions)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    authored_by: Mapped[str] = mapped_column(String, nullable=False, server_default="")
    mentioned_component_names: Mapped[list[dict]] = mapped_column(JSONB, nullable=False, default=list)
    mentioned_data_contract_names: Mapped[list[dict]] = mapped_column(JSONB, nullable=False, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class SilverClarification(Base):
    """History of every clarification question drafted and its answer, per source transcript.
    """

    __tablename__ = "silver_clarifications"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    tenant: Mapped[str] = mapped_column(String, nullable=False, index=True, server_default="default")
    ingestion_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    source_component: Mapped[str] = mapped_column(String, nullable=False, index=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    answered_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class SilverChunk(Base):
    """The single embedded chunk for one SilverDocument"""

    __tablename__ = "silver_chunks"
    __table_args__ = (
        UniqueConstraint("tenant", "source_component", "version", name="uq_silver_chunks_source_version"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    tenant: Mapped[str] = mapped_column(String, nullable=False, index=True, server_default="default")
    ingestion_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    source_component: Mapped[str] = mapped_column(String, nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(settings.embedding_dim), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class GoldEvolution(Base):
    """One asserted version of a component a data contract or the architecture"""

    __tablename__ = "gold_evolution"
    __table_args__ = (
        UniqueConstraint(
            "tenant", "entity_type", "entity_id", "version", name="uq_gold_evolution_entity_version"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    tenant: Mapped[str] = mapped_column(String, nullable=False, index=True, server_default="default")
    # This holds Literal["component", "data_contract", "architecture"]
    entity_type: Mapped[str] = mapped_column(String, nullable=False, index=True)
    entity_id: Mapped[str] = mapped_column(String, nullable=False)
    canonical_name: Mapped[str] = mapped_column(String, nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    operation: Mapped[str] = mapped_column(String, nullable=False)
    narrative: Mapped[str] = mapped_column(Text, nullable=False)
    # This holds ComponentPayload, DataContractPayload, or ArchitecturePayload
    # (agents.stages.gold.schemas). It is validated at the application layer before
    # being written here.
    payload: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    embedding: Mapped[list[float]] = mapped_column(Vector(settings.embedding_dim), nullable=False)
    entity_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    source_component: Mapped[str] = mapped_column(String, nullable=False, index=True)
    source_adr_version: Mapped[int] = mapped_column(Integer, nullable=False)
    authored_by: Mapped[str] = mapped_column(String, nullable=False, server_default="")
    ingestion_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    changed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    search_vector: Mapped[str | None] = mapped_column(
        TSVECTOR,
        Computed("to_tsvector('english', coalesce(canonical_name, '') || ' ' || coalesce(narrative, ''))"),
    )


class GoldAlias(Base):
    """Resolves different naming (aliases) for components and data contracts only"""

    __tablename__ = "gold_aliases"
    __table_args__ = (
        UniqueConstraint(
            "tenant", "entity_type", "entity_id", "alias", name="uq_gold_aliases_entity_alias"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    tenant: Mapped[str] = mapped_column(String, nullable=False, index=True, server_default="default")
    entity_type: Mapped[str] = mapped_column(String, nullable=False, index=True)
    entity_id: Mapped[str] = mapped_column(String, nullable=False)
    alias: Mapped[str] = mapped_column(String, nullable=False)
    source_component: Mapped[str] = mapped_column(String, nullable=False)
    source_adr_version: Mapped[int] = mapped_column(Integer, nullable=False)
    first_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class LlmCost(Base):
    """One row per real LLM call cost"""

    __tablename__ = "llm_costs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    tenant: Mapped[str] = mapped_column(String, nullable=False, index=True, server_default="default")
    method: Mapped[str] = mapped_column(String, nullable=False, index=True)
    model: Mapped[str] = mapped_column(String, nullable=False, index=True)
    prompt: Mapped[str] = mapped_column(Text, nullable=False)
    # EUR, via the same fixed `llm.router.USD_TO_EUR` rate
    input_cost: Mapped[float] = mapped_column(Float, nullable=False)
    output_cost: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)


