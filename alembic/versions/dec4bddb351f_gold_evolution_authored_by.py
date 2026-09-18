"""gold evolution authored by

`gold_evolution.authored_by` copies forward the same value `silver_documents.authored_by`
already holds for the ADR version this Gold row was extracted from (see
`fe437a248913_authorship_metadata_bronze_silver.py`). This follows the same "copy it onto the
row, do not join back to Silver" convention `gold_evolution.ingestion_date` already uses — see
that column's own comment in `db/models.py`. This is what lets an entity's evolution timeline
answer "who introduced it, and who changed it later" without a second query.

`server_default=''` keeps every row written before this migration valid.

Revision ID: dec4bddb351f
Revises: fe437a248913
Create Date: 2026-09-18 15:39:07.220158

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'dec4bddb351f'
down_revision: Union[str, Sequence[str], None] = 'fe437a248913'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "gold_evolution", sa.Column("authored_by", sa.String(), nullable=False, server_default="")
    )


def downgrade() -> None:
    op.drop_column("gold_evolution", "authored_by")
