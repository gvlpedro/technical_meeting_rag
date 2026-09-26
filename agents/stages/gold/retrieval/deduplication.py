"""TÉCNICA: Deduplicación por entidad en el top-k.

Qué problema resuelve: `gold_evolution` es un histórico "append-only" — cada nueva versión de un
componente o contrato es una fila NUEVA, nunca se sobrescribe la anterior (ver el docstring de
`GoldEvolution` en `db/models.py`). Eso significa que varias VERSIONES de la MISMA entidad suelen
tener narrativas casi idénticas entre sí, y en un top-k sin deduplicar pueden llegar a ocupar
varios de los huecos disponibles — desplazando a otra entidad genuinamente distinta y relevante
que nunca llega a aparecer en la respuesta.

Cómo funciona aquí: en vez de cortar directamente a los `k` mejores resultados, primero se pide
un conjunto de recall más amplio (`RECALL_POOL_SIZE` candidatos, definido en
`agents/stages/gold/service.py`), y `_dedupe_ids_by_entity` se queda con una sola fila por cada
`(entity_type, entity_id)` distinto — la VERSIÓN MÁS RECIENTE de esa entidad, no necesariamente
la fila que mejor puntuó en el ranking (una versión antigua puede, por azar, rankear más cerca de
la pregunta que la versión actual — resolverlo así respondería con un dato desactualizado a una
pregunta sobre el estado actual).

Quién la orquesta: `top_k_gold_evolution` la activa con `dedupe=True` (valor por defecto). Ver
`.tmp/advanced_techniques.md` §2 para la justificación completa."""

from sqlalchemy import and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import GoldEvolution


async def _dedupe_ids_by_entity(session: AsyncSession, ids: list[int], k: int) -> list[int]:
    """Collapses `ids` — already ranked, best first — down to at most `k` ids, one per distinct
    `(entity_type, entity_id)`. This runs in two passes, not one, because "keep whichever
    version ranked best" is the WRONG rule — a real regression this exact rule caused, caught
    by `agents.stages.gold.testing`'s real-LLM suite: an older version's narrative can rank
    closer to a question than the entity's own current state does (`top_k_gold_evolution`'s own
    docstring on `max_distance` already covers this for `latest_versions`' `[superseded]`
    tag — the same risk applies here). Deduping straight to that better-ranked OLDER version
    then answers "what changed" from stale state, missing exactly the update the question asked
    about.

    Pass 1 finds the RANK ORDER of the first `k` distinct entities in `ids` — this decides WHICH
    entities make the cut, and in what order, exactly as before. Pass 2 then resolves each of
    those entities to the id of its actual latest version — queried fresh from every version
    that entity has on record, not limited to whichever versions happened to be in `ids`. A
    version can be the entity's current state even if it ranked outside `ids` entirely (for
    example, past `RECALL_POOL_SIZE`), and this must still find it.

    Without deduping at all, several VERSIONS of the SAME entity — whose narratives are often
    near-identical between consecutive versions — can occupy multiple of the `k` slots a plain
    top-k would return, crowding out a genuinely different, relevant entity that never gets a
    chance to surface. See `RECALL_POOL_SIZE`'s own comment and `.tmp/advanced_techniques.md`
    §2."""
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
