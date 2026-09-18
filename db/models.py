from datetime import date, datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import Date, DateTime, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.config import settings


class Base(DeclarativeBase):
    pass


class BronzeDocument(Base):
    """Raw ingested transcript chunks. This is the Bronze layer of the medallion
    pipeline.

    The Silver layer (clarified, summary-chunked) and the Gold layer (structure-aware,
    component-graph) are future work. See the README's medallion diagram. Neither is
    built yet.
    """

    __tablename__ = "bronze_documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    # This is the frontend tenant that uploaded this row. It is a flat, copied-forward
    # column, the same no-FK convention as ingestion_date and source_component. It is
    # never joined against a users table, because there is no users table. Every query
    # anywhere in Bronze, Silver, or Gold must filter by tenant. Nothing here enforces
    # that at the DB level, beyond the unique constraints that include it.
    tenant: Mapped[str] = mapped_column(String, nullable=False, index=True, server_default="default")
    # This is the date the source video was uploaded on YouTube.
    ingestion_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    # This is the filename of the .vtt transcript this chunk was cut from. We keep it
    # for traceability.
    source_component: Mapped[str] = mapped_column(String, nullable=False)
    # This is who uploaded this chunk: the frontend's logged-in username (see
    # FrontendUser), passed through from `ingest_uploaded_files`. It is `""` for the
    # disk-based `ingest_bronze` path (`scripts/ingest.py` or `POST /v1/ingest`). That
    # path predates the frontend, so it has no user to attribute a row to.
    uploaded_by: Mapped[str] = mapped_column(String, nullable=False, server_default="")
    # This is the chunk's actual text, the text we feed to the embedding model.
    content: Mapped[str] = mapped_column(Text, nullable=False)
    # This is the vector embedding of `content`. Its dimension is fixed by
    # settings.embedding_dim and the configured sentence-transformers model. We use it
    # for similarity search.
    embedding: Mapped[list[float]] = mapped_column(Vector(settings.embedding_dim), nullable=False)
    # This is the row insertion timestamp: when this chunk was ingested, not when it
    # was recorded.
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class SilverDocument(Base):
    """One clarified ADR document per (source transcript, version). This is the Silver
    layer.

    The shape follows what `prompts/adr_generation/generator.jinja` produces. It is
    filled in as far as the clarification loop's answers allow. See
    `doc/silver_process.md` §1-2. `content` is the whole document, it is not chunked.
    `SilverChunk` below holds the single retrievable chunk cut from it. Silver does not
    store anything on disk. The clarification audit trail lives in
    `SilverClarification` below, not as a column here.

    This table is versioned, not just upserted. `write_document` hashes the freshly
    generated content and compares it against the latest existing row for this
    `source_component`. If the hash is identical, that same row is overwritten in
    place: same `version`, an idempotent re-run. If the hash is different, a new row is
    inserted at `version + 1`, and the older version's row is left untouched. Both stay
    queryable. `(source_component, version)` is the real identity now.
    `source_component` alone is no longer unique.

    `mentioned_component_names` and `mentioned_data_contract_names` store what
    `generate_architecture_questions` identifies for this row's own source. Before this
    column existed, that data was computed and then discarded every run. See the
    column comments below and `agents.shared.mentions_grounded_in_source`.
    """

    __tablename__ = "silver_documents"
    __table_args__ = (
        UniqueConstraint("tenant", "source_component", "version", name="uq_silver_documents_source_version"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    tenant: Mapped[str] = mapped_column(String, nullable=False, index=True, server_default="default")
    ingestion_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    # This matches bronze_documents.source_component. It is no longer unique alone, see
    # the class docstring. (source_component, version) is the real identity now.
    source_component: Mapped[str] = mapped_column(String, nullable=False, index=True)
    # This is 1, 2, 3, and so on, per source_component. It is bumped only when the
    # generated content's hash actually changes from the latest stored version.
    # write_document decides this, not the LLM.
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    # This is the sha256 hex digest of `content`. It is write_document's cheap way to
    # tell an "identical re-run" (overwrite this version) apart from "genuinely new
    # content" (new version), without diffing the full document text on every run.
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    # This is who generated THIS version. We extract it from `content`'s own
    # `**Authors:** <username>` line (`agents.graph._extract_authors_line`). We do not
    # pass it as a separate parameter. This way there is exactly one source of truth
    # for "who wrote this", and it can never drift from what the document itself says.
    # It is `""` when `content` has no Authors line at all, for example a CLI, script,
    # or test run with no real user (see `agents.state.SilverState.username`).
    authored_by: Mapped[str] = mapped_column(String, nullable=False, server_default="")
    # These are raw, unresolved component and data-contract mentions grounded in THIS
    # row's own source. They are `MentionedComponent` or `MentionedDataContract` shaped
    # dicts (agents/schemas.py). They are filtered from
    # generate_architecture_questions' batch-wide output down to just the ones whose
    # name appears verbatim in this source_component's transcript content (see
    # agents.shared.mentions_grounded_in_source). We call them "mentioned", not
    # "canonical", on purpose. They are not yet resolved through any identity or alias
    # matching. That resolution is Gold-level work. This column only feeds that work,
    # it does not do it.
    mentioned_component_names: Mapped[list[dict]] = mapped_column(JSONB, nullable=False, default=list)
    mentioned_data_contract_names: Mapped[list[dict]] = mapped_column(JSONB, nullable=False, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class SilverClarification(Base):
    """History of every clarification question drafted and its answer, per source
    transcript. This is the Silver layer, see `doc/silver_process.md` §2. This table is
    append-only. This is unlike `SilverDocument`, which upserts in place.
    `write_document` inserts a fresh row per (source_component, question) pair on
    every run, instead of overwriting a prior one. So if the same question gets
    answered differently across two runs, both answers stay on record, ordered by
    `answered_at`. `answer` is `NULL` for a question that was never actually answered:
    it was still `unknown` or `needs_clarification` at the time this row was written.
    """

    __tablename__ = "silver_clarifications"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    tenant: Mapped[str] = mapped_column(String, nullable=False, index=True, server_default="default")
    ingestion_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    # This matches bronze_documents.source_component and
    # silver_documents.source_component.
    source_component: Mapped[str] = mapped_column(String, nullable=False, index=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    answered_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class SilverChunk(Base):
    """The single embedded chunk for one SilverDocument (source_component, version).

    The rest of this pipeline splits text with a chunker, but an ADR is never split
    that way. One ADR is one chunk and one embedding. So a search hit always returns
    the whole decision record, not a fragment of it. There is no FK to
    silver_documents. This is the same flat source_component pattern bronze_documents
    already uses (doc/silver_process.md §2). But `(source_component, version)` is
    unique, so `chunk_and_embed` can upsert exactly the version it just wrote, without
    touching any other version's row.
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
    """One asserted version of a component, a data contract, or the architecture as a
    whole. This is the Gold layer, an append-only source of truth
    (`.tmp/gold_process.md` §6, cut to two tables and flattened per
    `.tmp/optmizaciones.md` §1 and `.tmp/refactor_silver_and_gold_process_v6.md`).

    There is no FK to `silver_documents`. `source_component` and
    `source_adr_version` are flat columns. This is the same no-FK convention
    `silver_chunks` already uses relative to `bronze_documents` (v6 §1).
    `ingestion_date` is copied forward too. This way, "how has X evolved over time"
    queries never need a join back to Silver just to get a date (v6 §2: this is
    event-time, while `changed_at` is processing-time).

    We use `entity_hash`, not `content_hash`. This is the same hash-compare-then-bump
    mechanism as `SilverDocument.content_hash`, but scoped to one entity's own
    `(operation, narrative, payload)`, not a whole document. A different column name
    avoids implying they hash the same shape of thing (v6 §4). "Current state" (the
    latest version per entity) is a query
    (`DISTINCT ON (entity_type, entity_id) ORDER BY version DESC`,
    `agents.stages.gold.service.current_gold_state`), not a materialized cache table.
    `v4`'s `gold_current_state` table is cut for this pass, see
    `.tmp/optmizaciones.md` §1.
    """

    __tablename__ = "gold_evolution"
    __table_args__ = (
        UniqueConstraint(
            "tenant", "entity_type", "entity_id", "version", name="uq_gold_evolution_entity_version"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    tenant: Mapped[str] = mapped_column(String, nullable=False, index=True, server_default="default")
    # This holds Literal["component", "data_contract", "architecture"], a subset of
    # agents.shared.QuestionScope. We do not redeclare that type here (v6 §3).
    # Pydantic enforces it at the application layer only. Here it is a plain `String`.
    # This is the same weak-DB-typing tradeoff already accepted for `operation` and
    # `payload` (gold_process.md §8).
    entity_type: Mapped[str] = mapped_column(String, nullable=False, index=True)
    # This is a stable identity, minted once by resolve_gold_identity. It is never
    # reused across entity_types.
    entity_id: Mapped[str] = mapped_column(String, nullable=False)
    canonical_name: Mapped[str] = mapped_column(String, nullable=False)
    # This is 1, 2, 3, and so on, per (entity_type, entity_id). This is a different
    # axis from SilverDocument.version, which works at whole-ADR granularity. Only the
    # hash-compare-then-bump *mechanism* is shared, not the counter itself.
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    # This holds ComponentStatus, ContractAction, or ArchitectureChangeType, depending
    # on entity_type. See agents.shared and agents.stages.gold.schemas; we do not
    # redeclare it here.
    operation: Mapped[str] = mapped_column(String, nullable=False)
    # This is the only embedded field. It holds clean prose for semantic search. It is
    # never a restatement of `payload`'s structured data (gold_process.md §1, v6 §5).
    narrative: Mapped[str] = mapped_column(Text, nullable=False)
    # This holds ComponentPayload, DataContractPayload, or ArchitecturePayload
    # (agents.stages.gold.schemas). It is validated at the application layer before
    # being written here.
    payload: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    embedding: Mapped[list[float]] = mapped_column(Vector(settings.embedding_dim), nullable=False)
    # This is sha256(operation, narrative, canonical-JSON payload). See
    # agents.stages.gold.service._entity_hash.
    entity_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    source_component: Mapped[str] = mapped_column(String, nullable=False, index=True)
    source_adr_version: Mapped[int] = mapped_column(Integer, nullable=False)
    # Who wrote the ADR version this row was extracted from. Copied forward from
    # `silver_documents.authored_by` at extraction time, the same "copy it onto the row"
    # convention `ingestion_date` above already follows, and for the same reason: "how has X
    # evolved over time, and who changed it" must never need a join back to Silver just to
    # name the author. `""` for a CLI, script, or test run with no real logged-in user, same
    # as `silver_documents.authored_by` and `bronze_documents.uploaded_by`.
    authored_by: Mapped[str] = mapped_column(String, nullable=False, server_default="")
    # This is event-time: the meeting date this version was asserted at, copied from
    # silver_documents.ingestion_date. It is not when this row was written, see
    # changed_at below.
    ingestion_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    changed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class GoldAlias(Base):
    """This is an identity-resolution lookup only (`gold_process.md` §3). It has one
    row per distinct raw name variant ever seen for an entity, not per version. So it
    stays small relative to `gold_evolution`, no matter how much history accumulates.
    It owns nothing: the `entity_id` values it resolves to live only as plain strings
    on `gold_evolution` rows. There is no FK relationship in either direction. This is
    the same flat, no-FK convention as `gold_evolution` itself (v6 §1).

    The resolution flow is: try an exact match on `alias`, then try a `pg_trgm` fuzzy
    match on the same column, then mint a new `entity_id`
    (`agents.stages.gold.service.resolve_entity_id`)."""

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


