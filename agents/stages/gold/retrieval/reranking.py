"""TÉCNICA: Reranking con cross-encoder.

Qué problema resuelve: un embedding compara la pregunta y cada narrativa como dos vectores
calculados POR SEPARADO — nunca "lee" ambos textos juntos. Eso puede dejar fuera del top-k una
fila que en realidad es la más relevante, simplemente porque su embedding no quedó lo bastante
cerca del de la pregunta en el espacio vectorial. Un cross-encoder, en cambio, lee la pregunta y
cada candidato JUNTOS en una sola pasada, y puede detectar relevancia que la comparación de dos
vectores por separado se pierde.

Cómo funciona aquí: sobre el conjunto ya deduplicado (pero todavía sin cortar a `k`),
`_rerank_ids` vuelve a puntuar cada candidato con un cross-encoder local
(`ingestion.reranker.score_candidates`, que corre en local, sin llamar a ningún proveedor
externo) leyendo `(pregunta, narrativa)` juntos, y ordena por esa nueva puntuación antes de
cortar. Es un paso opcional y deliberadamente apagado por defecto: cuesta una pasada de modelo
extra por candidato, y solo compensa cuando el ranking vectorial/híbrido deja fuera del top-k
algo que sí importaba.

Quién la orquesta: `top_k_gold_evolution` la activa con `rerank=True` (opt-in). Ver
`.tmp/advanced_techniques.md` §3 para la justificación completa, incluida la medición de coste
frente a beneficio que motivó dejarla apagada por defecto."""

import asyncio

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import GoldEvolution
from ingestion.reranker import score_candidates


async def _rerank_ids(session: AsyncSession, question_text: str, ids: list[int], k: int) -> list[int]:
    """Repunctuates `ids` — already ranked and deduped, but not yet cut to `k` — with a real
    local cross-encoder (`ingestion.reranker.score_candidates`), then keeps the best `k`. Unlike
    `_reciprocal_rank_fusion`, this does not combine rankings: it produces one new ranking,
    grounded in `(question_text, narrative)` pairs actually read together, and replaces
    whatever order `ids` arrived in.

    This queries `narrative` fresh for exactly the ids being reranked, not the full
    `GoldEvolution` row — a cross-encoder only ever reads the narrative text, and fetching less
    than the whole row keeps this step cheap relative to the model call itself, which already
    dominates its cost."""
    if not ids:
        return []
    rows = (
        await session.execute(select(GoldEvolution.id, GoldEvolution.narrative).where(GoldEvolution.id.in_(ids)))
    ).all()
    narrative_by_id = {row.id: row.narrative for row in rows}
    # An id from `ids` with no matching row here would mean it vanished between two queries in
    # the same call — should not happen, but skipping it is safer than crashing the whole
    # rerank over one stale id.
    present_ids = [item_id for item_id in ids if item_id in narrative_by_id]
    if not present_ids:
        return []

    scores = await asyncio.to_thread(score_candidates, question_text, [narrative_by_id[i] for i in present_ids])
    ranked = sorted(zip(present_ids, scores), key=lambda pair: pair[1], reverse=True)
    return [item_id for item_id, _ in ranked[:k]]
