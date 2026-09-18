"""gold_evolution, gold_aliases — Gold layer

This follows the Gold layer design in .tmp/gold_process.md v4. That design is cut down to two
tables and flattened, as described in .tmp/optmizaciones.md §1 and
.tmp/refactor_silver_and_gold_process_v6.md.

Revision ID: 22126087c3f7
Revises: eab9a1077fa9
Create Date: 2026-09-10 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

from app.config import settings

# revision identifiers, used by Alembic.
revision: str = '22126087c3f7'
down_revision: Union[str, Sequence[str], None] = 'eab9a1077fa9'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Neither table has a foreign key to silver_documents. Instead, each has flat
    (source_component, source_adr_version) columns. This follows the same no-FK convention
    that bronze and silver already use (see v6 §1).

    This migration enables pg_trgm for the first time in this schema, because only
    gold_aliases needs fuzzy matching. The `vector` extension was already enabled by the
    initial migration.
    """
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")

    op.create_table(
        "gold_evolution",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("entity_type", sa.String(), nullable=False),
        sa.Column("entity_id", sa.String(), nullable=False),
        sa.Column("canonical_name", sa.String(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("operation", sa.String(), nullable=False),
        sa.Column("narrative", sa.Text(), nullable=False),
        sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default="{}"),
        sa.Column("embedding", Vector(settings.embedding_dim), nullable=False),
        sa.Column("entity_hash", sa.String(64), nullable=False),
        sa.Column("source_component", sa.String(), nullable=False),
        sa.Column("source_adr_version", sa.Integer(), nullable=False),
        sa.Column("ingestion_date", sa.Date(), nullable=False),
        sa.Column("changed_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("entity_type", "entity_id", "version", name="uq_gold_evolution_entity_version"),
    )
    op.create_index("ix_gold_evolution_entity_type", "gold_evolution", ["entity_type"])
    op.create_index("ix_gold_evolution_source_component", "gold_evolution", ["source_component"])
    op.create_index("ix_gold_evolution_ingestion_date", "gold_evolution", ["ingestion_date"])
    op.create_index(
        "ix_gold_evolution_source", "gold_evolution", ["source_component", "source_adr_version"]
    )
    op.create_index(
        "ix_gold_evolution_payload_gin", "gold_evolution", ["payload"], postgresql_using="gin",
        postgresql_ops={"payload": "jsonb_path_ops"},
    )
    op.create_index(
        "ix_gold_evolution_embedding_hnsw", "gold_evolution", ["embedding"], postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )

    op.create_table(
        "gold_aliases",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("entity_type", sa.String(), nullable=False),
        sa.Column("entity_id", sa.String(), nullable=False),
        sa.Column("alias", sa.String(), nullable=False),
        sa.Column("source_component", sa.String(), nullable=False),
        sa.Column("source_adr_version", sa.Integer(), nullable=False),
        sa.Column("first_seen_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("entity_type", "entity_id", "alias", name="uq_gold_aliases_entity_alias"),
    )
    op.create_index("ix_gold_aliases_entity_type", "gold_aliases", ["entity_type"])
    op.create_index(
        "ix_gold_aliases_trgm", "gold_aliases", ["alias"], postgresql_using="gin",
        postgresql_ops={"alias": "gin_trgm_ops"},
    )


def downgrade() -> None:
    op.drop_index("ix_gold_aliases_trgm", table_name="gold_aliases")
    op.drop_index("ix_gold_aliases_entity_type", table_name="gold_aliases")
    op.drop_table("gold_aliases")

    op.drop_index("ix_gold_evolution_embedding_hnsw", table_name="gold_evolution")
    op.drop_index("ix_gold_evolution_payload_gin", table_name="gold_evolution")
    op.drop_index("ix_gold_evolution_source", table_name="gold_evolution")
    op.drop_index("ix_gold_evolution_ingestion_date", table_name="gold_evolution")
    op.drop_index("ix_gold_evolution_source_component", table_name="gold_evolution")
    op.drop_index("ix_gold_evolution_entity_type", table_name="gold_evolution")
    op.drop_table("gold_evolution")
