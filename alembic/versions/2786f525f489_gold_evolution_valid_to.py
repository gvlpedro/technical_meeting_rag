"""gold evolution valid_to

Adds `gold_evolution.valid_to`, nullable. NULL means "still the current version". The backfill
closes every historical row's `valid_to` to the `ingestion_date` of the version that superseded
it, using `LEAD()` over each `(tenant, entity_type, entity_id)` partition ordered by version —
the same window `persist_entity_version` now maintains going forward for every new row. See
`.tmp/improve_timeline_questions_and_linage.md` §2.1 and §6.

Revision ID: 2786f525f489
Revises: ae0262af77ef
Create Date: 2026-09-26 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '2786f525f489'
down_revision: Union[str, Sequence[str], None] = 'ae0262af77ef'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("gold_evolution", sa.Column("valid_to", sa.Date(), nullable=True))
    op.create_index("ix_gold_evolution_valid_to", "gold_evolution", ["valid_to"])
    op.execute(
        """
        WITH next_versions AS (
            SELECT
                id,
                LEAD(ingestion_date) OVER (
                    PARTITION BY tenant, entity_type, entity_id ORDER BY version
                ) AS next_ingestion_date
            FROM gold_evolution
        )
        UPDATE gold_evolution AS ge
        SET valid_to = nv.next_ingestion_date
        FROM next_versions AS nv
        WHERE ge.id = nv.id AND nv.next_ingestion_date IS NOT NULL
        """
    )


def downgrade() -> None:
    op.drop_index("ix_gold_evolution_valid_to", table_name="gold_evolution")
    op.drop_column("gold_evolution", "valid_to")
