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
    # Which frontend tenant uploaded this — a flat, copied-forward column, same no-FK
    # convention as ingestion_date/source_component (never joined against a users table,
    # there isn't one). Every query anywhere in Bronze/Silver/Gold must filter by tenant;
    # nothing here enforces that at the DB level beyond the unique constraints that include it.
    tenant: Mapped[str] = mapped_column(String, nullable=False, index=True, server_default="default")
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
    __table_args__ = (
        UniqueConstraint("tenant", "source_component", "version", name="uq_silver_documents_source_version"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    tenant: Mapped[str] = mapped_column(String, nullable=False, index=True, server_default="default")
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
    tenant: Mapped[str] = mapped_column(String, nullable=False, index=True, server_default="default")
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
    """One asserted version of a component, a data contract, or the architecture as a whole —
    Gold layer, append-only source of truth (`.tmp/gold_process.md` §6, cut to two tables and
    flattened per `.tmp/optmizaciones.md` §1 and `.tmp/refactor_silver_and_gold_process_v6.md`).

    No FK to `silver_documents` — `source_component`/`source_adr_version` are flat columns,
    same no-FK convention `silver_chunks` already uses relative to `bronze_documents`
    (v6 §1); `ingestion_date` is copied forward too, so "how has X evolved over time" queries
    never need a join back to Silver just to get a date (v6 §2, event-time vs. `changed_at`'s
    processing-time).

    `entity_hash`, not `content_hash`: same hash-compare-then-bump mechanism as
    `SilverDocument.content_hash`, but scoped to one entity's own
    `(operation, narrative, payload)`, not a whole document — a different column name avoids
    implying they hash the same shape of thing (v6 §4). "Current state" (latest version per
    entity) is a query (`DISTINCT ON (entity_type, entity_id) ORDER BY version DESC`,
    `agents.gold_service.current_gold_state`), not a materialized cache table — `v4`'s
    `gold_current_state` is cut for this pass, see `.tmp/optmizaciones.md` §1.
    """

    __tablename__ = "gold_evolution"
    __table_args__ = (
        UniqueConstraint(
            "tenant", "entity_type", "entity_id", "version", name="uq_gold_evolution_entity_version"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    tenant: Mapped[str] = mapped_column(String, nullable=False, index=True, server_default="default")
    # Literal["component", "data_contract", "architecture"] — a subset of
    # agents.schemas.QuestionScope, not redeclared (v6 §3); enforced by Pydantic at the
    # application layer only, plain `String` here, same weak-DB-typing tradeoff already
    # accepted for `operation`/`payload` (gold_process.md §8).
    entity_type: Mapped[str] = mapped_column(String, nullable=False, index=True)
    # Stable identity minted once by resolve_gold_identity, never reused across entity_types.
    entity_id: Mapped[str] = mapped_column(String, nullable=False)
    canonical_name: Mapped[str] = mapped_column(String, nullable=False)
    # Per (entity_type, entity_id): 1, 2, 3... — a different axis from
    # SilverDocument.version (whole-ADR granularity); only the hash-compare-then-bump
    # *mechanism* is shared, not the counter itself.
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    # ComponentStatus | ContractAction | ArchitectureChangeType depending on entity_type —
    # agents.schemas, not redeclared here.
    operation: Mapped[str] = mapped_column(String, nullable=False)
    # The only embedded field — clean prose for semantic search, never a restatement of
    # `payload`'s structured data (gold_process.md §1, v6 §5).
    narrative: Mapped[str] = mapped_column(Text, nullable=False)
    # ComponentPayload | DataContractPayload | ArchitecturePayload (agents.schemas),
    # validated at the application layer before being written here.
    payload: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    embedding: Mapped[list[float]] = mapped_column(Vector(settings.embedding_dim), nullable=False)
    # sha256(operation, narrative, canonical-JSON payload) — see agents.gold_service._entity_hash.
    entity_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    source_component: Mapped[str] = mapped_column(String, nullable=False, index=True)
    source_adr_version: Mapped[int] = mapped_column(Integer, nullable=False)
    # Event-time: the meeting date this version was asserted at (copied from
    # silver_documents.ingestion_date), not when this row was written — see changed_at below.
    ingestion_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    changed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class GoldAlias(Base):
    """Identity-resolution lookup only (`gold_process.md` §3) — one row per distinct raw name
    variant ever seen for an entity, not per version, so it stays small relative to
    `gold_evolution` regardless of how much history accumulates. Owns nothing: the
    `entity_id` values it resolves to live only as plain strings on `gold_evolution` rows, no
    FK relationship in either direction (same flat, no-FK convention as `gold_evolution`
    itself, v6 §1).

    Resolution flow: exact match on `alias` -> `pg_trgm` fuzzy match on the same column ->
    mint a new `entity_id` (`agents.gold_service.resolve_entity_id`)."""

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


