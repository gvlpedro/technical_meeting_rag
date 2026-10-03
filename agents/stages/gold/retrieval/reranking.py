"""TECHNIQUE: Cross-encoder reranking.

Problem: an embedding compares the question and each narrative as two separately-computed
vectors. It never reads both together, so the truly best row can miss the top-k anyway.

How: on the already-deduped set (not yet cut to `k`), `_rerank_ids` re-scores each candidate
with a local cross-encoder (`ingestion.reranker.score_candidates`), reading `(question,
narrative)` together, then sorts by that score before cutting. Costs one extra model pass per
candidate, so this is off by default.

Used by: `top_k_gold_evolution`, with `rerank=True` (opt-in)."""

import asyncio

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import GoldEvolution
from ingestion.reranker import score_candidates


async def _rerank_ids(session: AsyncSession, question_text: str, ids: list[int], k: int) -> list[int]:
    """Re-scores `ids` with a local cross-encoder, reading `(question_text, narrative)`
    together, then keeps the best `k`. Replaces the input order with one new ranking; unlike
    `_reciprocal_rank_fusion`, it does not combine rankings.

    Queries only `narrative` for these ids, not the full row, since the cross-encoder reads
    narrative text only."""
    if not ids:
        return []
    rows = (
        await session.execute(select(GoldEvolution.id, GoldEvolution.narrative).where(GoldEvolution.id.in_(ids)))
    ).all()
    narrative_by_id = {row.id: row.narrative for row in rows}
    # Skip an id with no matching row instead of crashing the whole rerank.
    present_ids = [item_id for item_id in ids if item_id in narrative_by_id]
    if not present_ids:
        return []

    scores = await asyncio.to_thread(score_candidates, question_text, [narrative_by_id[i] for i in present_ids])
    ranked = sorted(zip(present_ids, scores), key=lambda pair: pair[1], reverse=True)
    return [item_id for item_id, _ in ranked[:k]]
