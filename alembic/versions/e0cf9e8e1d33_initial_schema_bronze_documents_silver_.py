"""initial schema: bronze_documents, silver_documents, silver_clarifications, silver_chunks

Revision ID: e0cf9e8e1d33
Revises: 
Create Date: 2026-09-08 17:15:31.403368

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

from app.config import settings

# revision identifiers, used by Alembic.
revision: str = 'e0cf9e8e1d33'
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

    op.create_table(
        "silver_documents",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("ingestion_date", sa.Date(), nullable=False),
        sa.Column("source_component", sa.String(), nullable=False, unique=True),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_silver_documents_ingestion_date", "silver_documents", ["ingestion_date"])

    op.create_table(
        "silver_clarifications",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("ingestion_date", sa.Date(), nullable=False),
        sa.Column("source_component", sa.String(), nullable=False),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("answer", sa.Text(), nullable=True),
        sa.Column("answered_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index(
        "ix_silver_clarifications_ingestion_date", "silver_clarifications", ["ingestion_date"]
    )
    op.create_index(
        "ix_silver_clarifications_source_component", "silver_clarifications", ["source_component"]
    )

    op.create_table(
        "silver_chunks",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("ingestion_date", sa.Date(), nullable=False),
        sa.Column("source_component", sa.String(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(settings.embedding_dim), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_silver_chunks_ingestion_date", "silver_chunks", ["ingestion_date"])


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_silver_chunks_ingestion_date", table_name="silver_chunks")
    op.drop_table("silver_chunks")

    op.drop_index("ix_silver_clarifications_source_component", table_name="silver_clarifications")
    op.drop_index("ix_silver_clarifications_ingestion_date", table_name="silver_clarifications")
    op.drop_table("silver_clarifications")

    op.drop_index("ix_silver_documents_ingestion_date", table_name="silver_documents")
    op.drop_table("silver_documents")

    op.drop_index("ix_bronze_documents_ingestion_date", table_name="bronze_documents")
    op.drop_table("bronze_documents")
