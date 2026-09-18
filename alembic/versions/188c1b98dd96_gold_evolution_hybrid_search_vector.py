"""gold evolution hybrid search vector

`gold_evolution.search_vector` is a Postgres GENERATED ALWAYS AS ... STORED column, combining
`canonical_name` and `narrative` into one `tsvector`, with a GIN index. Postgres recomputes it
on every INSERT/UPDATE — the application never writes to it directly (see `db/models.py`'s
`GoldEvolution.search_vector`, declared with `sa.Computed(...)` so the ORM never includes it in
an INSERT/UPDATE statement either).

This backs `agents.stages.gold.service.top_k_gold_evolution`'s new hybrid-search branch: a
lexical `ts_rank` ranking, fused with the existing vector cosine-distance ranking via Reciprocal
Rank Fusion. See that function's own docstring for why a name/acronym/jargon match a plain
embedding blurs is worth a second, independent ranking signal.

`to_tsvector('english', ...)` matches the observed language of every `narrative` this project
has generated so far (see `prompts/gold/extraction.jinja`) — English stemming/stopword removal,
not a language-neutral tokenizer. `coalesce(..., '')` guards against a NULL `canonical_name`
(never happens today, `nullable=False`) still not blowing up string concatenation.

Revision ID: 188c1b98dd96
Revises: dec4bddb351f
Create Date: 2026-09-18 18:22:52.401168

"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = '188c1b98dd96'
down_revision: Union[str, Sequence[str], None] = 'dec4bddb351f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE gold_evolution
        ADD COLUMN search_vector tsvector
        GENERATED ALWAYS AS (
            to_tsvector('english', coalesce(canonical_name, '') || ' ' || coalesce(narrative, ''))
        ) STORED
        """
    )
    op.execute("CREATE INDEX ix_gold_evolution_search_vector ON gold_evolution USING gin (search_vector)")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_gold_evolution_search_vector")
    op.execute("ALTER TABLE gold_evolution DROP COLUMN IF EXISTS search_vector")
