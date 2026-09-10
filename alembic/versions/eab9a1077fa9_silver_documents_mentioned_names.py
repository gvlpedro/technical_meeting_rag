"""silver_documents mentioned_component_names/mentioned_data_contract_names

Revision ID: eab9a1077fa9
Revises: 08baba40461e
Create Date: 2026-09-10 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'eab9a1077fa9'
down_revision: Union[str, Sequence[str], None] = '08baba40461e'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Persists what `generate_architecture_questions` already computes per ingestion_date
    batch (mentioned components/data contracts, grounded per source via
    `agents.service.mentions_grounded_in_source`) instead of discarding it once the graph
    run ends. Backfilled to `'[]'` for existing rows — this data was never computed for
    past runs, not a value that can be reconstructed from `content` alone.
    """
    op.add_column(
        "silver_documents",
        sa.Column(
            "mentioned_component_names",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default="[]",
        ),
    )
    op.add_column(
        "silver_documents",
        sa.Column(
            "mentioned_data_contract_names",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default="[]",
        ),
    )


def downgrade() -> None:
    op.drop_column("silver_documents", "mentioned_data_contract_names")
    op.drop_column("silver_documents", "mentioned_component_names")
