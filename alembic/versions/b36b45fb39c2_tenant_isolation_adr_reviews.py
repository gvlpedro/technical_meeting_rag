"""tenant isolation, adr_reviews

Every table Bronze/Silver/Gold write to gets a flat `tenant` column, same no-FK,
copied-forward convention `ingestion_date`/`source_component` already use — a query that
forgets to filter by tenant is a bug, not a missing join. `server_default='default'` so
every row written before the frontend existed (and every existing test that never passes a
tenant) keeps working unchanged, under one implicit "default" tenant.

`adr_reviews` is a new, separate table (not a column on `silver_documents`) so accepting an
ADR never touches the Silver/Gold write path at all — Gold still extracts automatically the
moment `write_document` runs, exactly as before this migration. The Chat tab is the only
thing that reads `adr_reviews`, to decide whether a (tenant, source_component, version) is
approved yet.

Revision ID: b36b45fb39c2
Revises: 22126087c3f7
Create Date: 2026-09-15 13:24:15.048738

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b36b45fb39c2'
down_revision: Union[str, Sequence[str], None] = '22126087c3f7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_TENANTED_TABLES = [
    "bronze_documents",
    "silver_documents",
    "silver_clarifications",
    "silver_chunks",
    "gold_evolution",
    "gold_aliases",
]


def upgrade() -> None:
    for table in _TENANTED_TABLES:
        op.add_column(
            table, sa.Column("tenant", sa.String(), nullable=False, server_default="default")
        )
        op.create_index(f"ix_{table}_tenant", table, ["tenant"])

    op.drop_constraint("uq_silver_documents_source_version", "silver_documents", type_="unique")
    op.create_unique_constraint(
        "uq_silver_documents_source_version", "silver_documents", ["tenant", "source_component", "version"]
    )

    op.drop_constraint("uq_silver_chunks_source_version", "silver_chunks", type_="unique")
    op.create_unique_constraint(
        "uq_silver_chunks_source_version", "silver_chunks", ["tenant", "source_component", "version"]
    )

    op.drop_constraint("uq_gold_evolution_entity_version", "gold_evolution", type_="unique")
    op.create_unique_constraint(
        "uq_gold_evolution_entity_version",
        "gold_evolution",
        ["tenant", "entity_type", "entity_id", "version"],
    )

    op.drop_constraint("uq_gold_aliases_entity_alias", "gold_aliases", type_="unique")
    op.create_unique_constraint(
        "uq_gold_aliases_entity_alias", "gold_aliases", ["tenant", "entity_type", "entity_id", "alias"]
    )

    op.create_table(
        "adr_reviews",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("tenant", sa.String(), nullable=False),
        sa.Column("source_component", sa.String(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("reviewed_by", sa.String(), nullable=False),
        sa.Column("reviewed_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("tenant", "source_component", "version", name="uq_adr_reviews_tenant_source_version"),
    )
    op.create_index("ix_adr_reviews_tenant", "adr_reviews", ["tenant"])


def downgrade() -> None:
    op.drop_index("ix_adr_reviews_tenant", table_name="adr_reviews")
    op.drop_table("adr_reviews")

    op.drop_constraint("uq_gold_aliases_entity_alias", "gold_aliases", type_="unique")
    op.create_unique_constraint(
        "uq_gold_aliases_entity_alias", "gold_aliases", ["entity_type", "entity_id", "alias"]
    )

    op.drop_constraint("uq_gold_evolution_entity_version", "gold_evolution", type_="unique")
    op.create_unique_constraint(
        "uq_gold_evolution_entity_version", "gold_evolution", ["entity_type", "entity_id", "version"]
    )

    op.drop_constraint("uq_silver_chunks_source_version", "silver_chunks", type_="unique")
    op.create_unique_constraint(
        "uq_silver_chunks_source_version", "silver_chunks", ["source_component", "version"]
    )

    op.drop_constraint("uq_silver_documents_source_version", "silver_documents", type_="unique")
    op.create_unique_constraint(
        "uq_silver_documents_source_version", "silver_documents", ["source_component", "version"]
    )

    for table in _TENANTED_TABLES:
        op.drop_index(f"ix_{table}_tenant", table_name=table)
        op.drop_column(table, "tenant")
