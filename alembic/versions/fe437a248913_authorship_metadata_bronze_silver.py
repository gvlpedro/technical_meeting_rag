"""authorship metadata bronze silver

`bronze_documents.uploaded_by` stores who uploaded this chunk: the frontend's logged-in
username. `silver_documents.authored_by` stores who authored this published version. This
value is extracted from the ADR content's own `**Authors:**` line at persist time. It is never
a second, independent value that could drift from what the document itself says.

Both columns use `server_default=''`. This keeps every row written before this migration
valid, and also every disk-based `ingest_bronze` row, which has no user to attribute.

Revision ID: fe437a248913
Revises: c071da0da7f2
Create Date: 2026-09-16 13:11:56.419237

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'fe437a248913'
down_revision: Union[str, Sequence[str], None] = 'c071da0da7f2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "bronze_documents", sa.Column("uploaded_by", sa.String(), nullable=False, server_default="")
    )
    op.add_column(
        "silver_documents", sa.Column("authored_by", sa.String(), nullable=False, server_default="")
    )


def downgrade() -> None:
    op.drop_column("silver_documents", "authored_by")
    op.drop_column("bronze_documents", "uploaded_by")
