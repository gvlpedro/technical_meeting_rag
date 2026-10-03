"""TECHNIQUE: Per-entity deduplication in the top-k.

Problem: `gold_evolution` keeps every version as its own row. Versions of the same entity
often have near-identical narratives. Without dedup, they can fill several top-k slots and
crowd out a different, relevant entity.

How: `top_k_gold_evolution` first fetches a wider pool (`RECALL_POOL_SIZE` candidates).
`_dedupe_ids_by_entity` then keeps one row per `(entity_type, entity_id)` — its latest
version, not necessarily the best-ranked row. An old version can rank closer to the question
by chance; answering from it would give stale state for a question about the current one.

Used by: `top_k_gold_evolution`, with `dedupe=True` (the default)."""

from sqlalchemy import and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import GoldEvolution


async def _dedupe_ids_by_entity(session: AsyncSession, ids: list[int], k: int) -> list[int]:
    """Collapses `ids` — already ranked, best first — to at most `k` ids, one per distinct
    `(entity_type, entity_id)`. Two passes: pass 1 picks the rank order of the first `k`
    distinct entities. Pass 2 resolves each entity to its actual latest version, queried fresh,
    not limited to the versions already in `ids`. Keeping "whichever version ranked best"
    instead is wrong: an older version can rank closer to the question than the current one,
    which answers from stale state."""
    if not ids:
        return []
    candidate_rows = (
        await session.execute(
            select(GoldEvolution.id, GoldEvolution.entity_type, GoldEvolution.entity_id).where(
                GoldEvolution.id.in_(ids)
            )
        )
    ).all()
    entity_by_id = {row.id: (row.entity_type, row.entity_id) for row in candidate_rows}

    ordered_entities: list[tuple[str, str]] = []
    seen_entities: set[tuple[str, str]] = set()
    for item_id in ids:
        entity_key = entity_by_id.get(item_id)
        if entity_key is None or entity_key in seen_entities:
            continue
        seen_entities.add(entity_key)
        ordered_entities.append(entity_key)
        if len(ordered_entities) == k:
            break
    if not ordered_entities:
        return []

    every_version = (
        await session.execute(
            select(GoldEvolution.id, GoldEvolution.entity_type, GoldEvolution.entity_id, GoldEvolution.version).where(
                or_(
                    *(
                        and_(GoldEvolution.entity_type == entity_type, GoldEvolution.entity_id == entity_id)
                        for entity_type, entity_id in ordered_entities
                    )
                )
            )
        )
    ).all()
    latest_by_entity: dict[tuple[str, str], tuple[int, int]] = {}  # entity -> (version, id)
    for row in every_version:
        entity_key = (row.entity_type, row.entity_id)
        current_best = latest_by_entity.get(entity_key)
        if current_best is None or row.version > current_best[0]:
            latest_by_entity[entity_key] = (row.version, row.id)

    return [latest_by_entity[entity_key][1] for entity_key in ordered_entities]
