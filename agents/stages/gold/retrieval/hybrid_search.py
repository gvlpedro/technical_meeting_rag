"""TECHNIQUE: Hybrid search (vector + lexical, fused with Reciprocal Rank Fusion).

Problem: a pure vector search can blur an exact match — a proper noun, an acronym — because
the embedding captures general meaning, not exact words. Lexical search finds exact matches
but misses a question phrased with different words for the same meaning.

How: two independent searches run over `gold_evolution` — by cosine distance
(`_top_k_ids_by_vector`) and by `ts_rank` over the GIN-indexed `search_vector` column
(`_top_k_ids_by_lexical_rank`). `_reciprocal_rank_fusion` merges both rankings: each row scores
`1 / (k_constant + rank)` per ranking it appears in, summed. A row that is the best lexical
match but a weak vector match can still win the fusion.

Used by: `top_k_gold_evolution`, with `mode="hybrid"`."""

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import GoldEvolution

# Reciprocal Rank Fusion's smoothing constant. 60 is the original RRF paper's tuned value
# (Cormack et al., 2009) and the common default. A larger value flattens rank gaps; a smaller
# one lets rank 1 dominate. Not yet tuned against this corpus.
RRF_K_CONSTANT = 60


def _reciprocal_rank_fusion(rankings: list[list[int]], k_constant: int = RRF_K_CONSTANT) -> list[int]:
    """Fuses ranked id lists into one ranking. Each id's score is the sum, over every input
    ranking it appears in, of `1 / (k_constant + rank)`. An id missing from a ranking gets no
    penalty from it. A row only needs to rank well in ONE signal to surface near the top.

    Returns ids sorted by fused score, descending."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, item_id in enumerate(ranking, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k_constant + rank)
    return sorted(scores, key=lambda item_id: scores[item_id], reverse=True)


async def _top_k_ids_by_vector(
    session: AsyncSession,
    vector: list[float],
    limit: int,
    source_component: str | None,
    max_distance: float | None,
    tenant: str,
) -> list[int]:
    distance = GoldEvolution.embedding.cosine_distance(vector)
    query = select(GoldEvolution.id).where(GoldEvolution.tenant == tenant).order_by(distance).limit(limit)
    if source_component is not None:
        query = query.where(GoldEvolution.source_component == source_component)
    if max_distance is not None:
        query = query.where(distance <= max_distance)
    return list((await session.execute(query)).scalars().all())


async def _top_k_ids_by_lexical_rank(
    session: AsyncSession,
    question_text: str,
    limit: int,
    source_component: str | None,
    tenant: str,
) -> list[int]:
    """The lexical half of hybrid search: Postgres full-text search over `search_vector`,
    ranked by `ts_rank`. `plainto_tsquery` treats `question_text` as plain text, not `tsquery`
    syntax, so `&`/`|`/`!` in a question are not read as search operators. A question with no
    recognized lexemes matches nothing here; RRF then falls back to the vector ranking alone."""
    tsquery = func.plainto_tsquery("english", question_text)
    query = (
        select(GoldEvolution.id)
        .where(GoldEvolution.tenant == tenant, GoldEvolution.search_vector.op("@@")(tsquery))
        .order_by(func.ts_rank(GoldEvolution.search_vector, tsquery).desc())
        .limit(limit)
    )
    if source_component is not None:
        query = query.where(GoldEvolution.source_component == source_component)
    return list((await session.execute(query)).scalars().all())
