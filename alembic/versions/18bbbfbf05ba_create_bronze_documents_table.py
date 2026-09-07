"""create bronze_documents table

Revision ID: 18bbbfbf05ba
Revises: 
Create Date: 2026-09-07 09:38:00.391477

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

from app.config import settings

# revision identifiers, used by Alembic.
revision: str = '18bbbfbf05ba'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "bronze_documents",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("ingestion_date", sa.Date(), nullable=False),
        sa.Column("source_component", sa.String(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(settings.embedding_dim), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_bronze_documents_ingestion_date", "bronze_documents", ["ingestion_date"])


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_bronze_documents_ingestion_date", table_name="bronze_documents")
    op.drop_table("bronze_documents")
