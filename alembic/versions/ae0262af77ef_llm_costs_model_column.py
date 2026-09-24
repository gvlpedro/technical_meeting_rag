"""llm_costs model column

Revision ID: ae0262af77ef
Revises: 5028f8fbf80f
Create Date: 2026-09-23 15:01:48.911067

Hand-trimmed the same way `5028f8fbf80f` was — `--autogenerate` picked up the same unrelated
drift (LangGraph's `checkpoint*` tables, raw-SQL indexes) that migration's own docstring
already explains. Only the one column this change actually adds is kept here.

`server_default='unknown'` exists only so `ADD COLUMN ... NOT NULL` does not fail against the
table's own already-existing rows — every future row from `llm.router.complete` always passes
a real `model` value explicitly, so this default is a one-time backfill value, never a value
the application code relies on.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'ae0262af77ef'
down_revision: Union[str, Sequence[str], None] = '5028f8fbf80f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('llm_costs', sa.Column('model', sa.String(), server_default='unknown', nullable=False))
    op.create_index(op.f('ix_llm_costs_model'), 'llm_costs', ['model'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_llm_costs_model'), table_name='llm_costs')
    op.drop_column('llm_costs', 'model')
