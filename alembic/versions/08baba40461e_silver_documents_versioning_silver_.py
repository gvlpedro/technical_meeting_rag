"""silver_documents versioning, silver_chunks single-chunk-per-version

Revision ID: 08baba40461e
Revises: e0cf9e8e1d33
Create Date: 2026-09-09 10:13:57.361726

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '08baba40461e'
down_revision: Union[str, Sequence[str], None] = 'e0cf9e8e1d33'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema.

    silver_documents/silver_chunks change shape here: synthesize_document now writes
    ADR-shaped content (prompts/roles/common/adr_generator.jinja) instead of
    clarification_template.md's shape, silver_documents gains (source_component,
    version) identity instead of source_component alone, and silver_chunks moves from
    many token-split chunks per document to exactly one chunk per document version.
    Both tables are fully re-derivable from bronze_documents by re-running
    `make clarify` for the affected ingestion dates, so existing rows are truncated
    here rather than backfilled into a shape they were never generated in.
    bronze_documents and silver_clarifications (append-only history) are untouched.
    """
    op.execute("TRUNCATE TABLE silver_chunks")
    op.execute("TRUNCATE TABLE silver_documents")

    op.drop_constraint("silver_documents_source_component_key", "silver_documents", type_="unique")
    op.add_column("silver_documents", sa.Column("version", sa.Integer(), nullable=False))
    op.add_column("silver_documents", sa.Column("content_hash", sa.String(length=64), nullable=False))
    op.create_index("ix_silver_documents_source_component", "silver_documents", ["source_component"])
    op.create_unique_constraint(
        "uq_silver_documents_source_version", "silver_documents", ["source_component", "version"]
    )

    op.add_column("silver_chunks", sa.Column("version", sa.Integer(), nullable=False))
    op.create_unique_constraint(
        "uq_silver_chunks_source_version", "silver_chunks", ["source_component", "version"]
    )


def downgrade() -> None:
    """Downgrade schema. Destructive, like the migration it reverses: existing
    versioned rows have no single "current" row to collapse back into, so this drops
    the new columns/constraints rather than attempting to reconstruct the old shape.
    """
    op.drop_constraint("uq_silver_chunks_source_version", "silver_chunks", type_="unique")
    op.drop_column("silver_chunks", "version")

    op.drop_constraint("uq_silver_documents_source_version", "silver_documents", type_="unique")
    op.drop_index("ix_silver_documents_source_component", table_name="silver_documents")
    op.drop_column("silver_documents", "content_hash")
    op.drop_column("silver_documents", "version")
    op.create_unique_constraint(
        "silver_documents_source_component_key", "silver_documents", ["source_component"]
    )
