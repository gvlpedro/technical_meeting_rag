"""llm_costs table

Revision ID: 5028f8fbf80f
Revises: 188c1b98dd96
Create Date: 2026-09-23 14:54:53.732945

`--autogenerate` also picked up a pile of unrelated drift between the live DB and what
`db/models.py` declares (LangGraph's own `checkpoint*` tables, which it manages itself and
were never in `Base.metadata`; several indexes created via raw `op.execute` SQL in earlier
migrations, like `gold_evolution`'s HNSW/GIN/search_vector indexes, that autogenerate's
column-level diffing does not recognize as already matching). None of that belongs in this
migration — this file was hand-trimmed down to just the one table this change actually adds.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '5028f8fbf80f'
down_revision: Union[str, Sequence[str], None] = '188c1b98dd96'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'llm_costs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('tenant', sa.String(), server_default='default', nullable=False),
        sa.Column('method', sa.String(), nullable=False),
        sa.Column('prompt', sa.Text(), nullable=False),
        sa.Column('input_cost', sa.Float(), nullable=False),
        sa.Column('output_cost', sa.Float(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_llm_costs_created_at'), 'llm_costs', ['created_at'], unique=False)
    op.create_index(op.f('ix_llm_costs_method'), 'llm_costs', ['method'], unique=False)
    op.create_index(op.f('ix_llm_costs_tenant'), 'llm_costs', ['tenant'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_llm_costs_tenant'), table_name='llm_costs')
    op.drop_index(op.f('ix_llm_costs_method'), table_name='llm_costs')
    op.drop_index(op.f('ix_llm_costs_created_at'), table_name='llm_costs')
    op.drop_table('llm_costs')
