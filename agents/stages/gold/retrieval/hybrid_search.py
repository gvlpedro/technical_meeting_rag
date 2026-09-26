"""TÉCNICA: Búsqueda híbrida (vectorial + léxica, fusionadas con Reciprocal Rank Fusion).

Qué problema resuelve: una búsqueda puramente vectorial (por similitud de embeddings) puede
"difuminar" una coincidencia exacta — el nombre propio de un componente, un acrónimo, un campo
concreto de un ODCS — porque el embedding representa el SIGNIFICADO general del texto, no sus
palabras exactas. La búsqueda léxica (full-text search de Postgres, por palabras) sí encuentra
esas coincidencias exactas, pero se le escapa todo lo que se pregunta con otras palabras aunque
signifique lo mismo.

Cómo funciona aquí: se lanzan dos búsquedas independientes sobre `gold_evolution` — una por
distancia coseno del embedding (`_top_k_ids_by_vector`), y otra por `ts_rank` sobre la columna
`search_vector` ya indexada con GIN (`_top_k_ids_by_lexical_rank`). Los dos rankings resultantes
se fusionan con Reciprocal Rank Fusion (`_reciprocal_rank_fusion`): cada fila puntúa por
`1 / (k_constant + puesto)` en cada ranking donde aparezca, y esas puntuaciones se suman — así
una fila que sea el mejor resultado léxico pero un resultado mediocre en el vectorial (el caso
típico de un nombre propio) puede seguir ganando en la fusión, sin necesitar aparecer arriba en
ambas búsquedas a la vez.

Quién la orquesta: `top_k_gold_evolution` (en `agents/stages/gold/service.py`) activa este modo
con `mode="hybrid"`. Ver `.tmp/advanced_techniques.md` §1 para la justificación completa."""

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import GoldEvolution

# Reciprocal Rank Fusion's own smoothing constant. 60 is the value the original RRF paper (Cormack
# et al., 2009) tuned against, and it is what most hybrid-search implementations still default
# to — a larger value flattens the gap between rank 1 and rank 30, a smaller one makes rank 1
# dominate almost completely. Nothing about this corpus has been measured against a different
# value yet, so this stays at the well-established default rather than an invented number.
RRF_K_CONSTANT = 60


def _reciprocal_rank_fusion(rankings: list[list[int]], k_constant: int = RRF_K_CONSTANT) -> list[int]:
    """Fuses any number of ranked id lists into one combined ranking. Each id's fused score is
    the sum, across every input ranking it appears in, of `1 / (k_constant + rank)` (rank
    counted from 1). An id absent from one of the rankings contributes nothing from it — not a
    penalty. That is the entire point of fusing instead of intersecting: a row only needs to
    rank well in ONE of the two signals (semantic OR lexical) to surface near the top, since a
    row that is the single best lexical match but a mediocre embedding match (an exact proper
    noun, an acronym) is exactly the case hybrid search exists to rescue.

    Returns ids sorted by fused score, descending. Ties (an id with the same fused score as
    another, which only realistically happens for two ids appearing in neither ranking) keep
    Python's stable sort order — the order they were first seen in `rankings`."""
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
    """The lexical half of hybrid search: Postgres full-text search over `search_vector` (see
    that column's own comment in `db/models.py` and migration `188c1b98dd96`), ranked by
    `ts_rank`. `plainto_tsquery` treats `question_text` as plain text, not `tsquery` syntax — a
    question typed by a person is not a search-operator expression, and letting `&`/`|`/`!`
    characters in a question be interpreted as tsquery operators would be a second, unrelated
    injection surface. A question that reduces to an empty tsquery (all stopwords, or no
    recognized lexemes) matches nothing here — RRF then falls back to whatever the vector
    ranking alone found, which is the correct degrade: no lexical signal is not an error."""
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
