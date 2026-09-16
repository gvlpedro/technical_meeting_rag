"""drop adr_reviews

The "Pull request" tab (accept/reject one ADR before Chat could use it) is gone — replaced by
"Architecture history", a read-only view. Publishing an ADR now means one thing: persisted to
Gold, immediately retrievable, no separate approval gate. `adr_reviews` had exactly one reader
(the Chat tab's own gate, since removed from `app/routers/frontend.py`), so nothing else in
Bronze/Silver/Gold depends on this table existing.

Revision ID: c071da0da7f2
Revises: b36b45fb39c2
Create Date: 2026-09-16 09:34:23.872477

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c071da0da7f2'
down_revision: Union[str, Sequence[str], None] = 'b36b45fb39c2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_index("ix_adr_reviews_tenant", table_name="adr_reviews")
    op.drop_table("adr_reviews")


def downgrade() -> None:
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
