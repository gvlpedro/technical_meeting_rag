"""TÉCNICA: Corrective RAG (reintento automático con un filtro más laxo).

Qué problema resuelve: una búsqueda con un `max_distance` estricto (ver
`DEFAULT_MAX_DISTANCE` en `agents/stages/gold/service.py`) existe para no forzar una respuesta a
partir del candidato "menos malo" cuando en realidad nada es lo bastante relevante — pero ese
mismo filtro puede dejar sin respuesta una pregunta que sí tenía un hecho relevante, solo que su
distancia coseno quedó un poco por encima del corte. Sin este reintento, esas preguntas
terminan siempre en "no sé", aunque el dato exista.

Cómo funciona aquí: `retrieve_with_correction` no sabe nada de Gold, de Postgres ni de
embeddings — recibe una función `retrieve` que acepta un `max_distance` y devuelve filas, y la
llama primero con el filtro estricto. Si esa primera llamada no devuelve nada, la vuelve a
llamar una única vez con un filtro más laxo (por defecto, sin filtro en absoluto) antes de
rendirse. Ser agnóstico a Gold es deliberado: así se puede probar con una función de mentira en
un test rápido, sin tocar la base de datos, y reutilizar para cualquier otra búsqueda que en el
futuro necesite el mismo patrón de "reintenta una vez, más laxo, antes de rendirte".

Un solo reintento, nunca más: si la segunda llamada tampoco encuentra nada, la pregunta
realmente no tiene un hecho relevante que ofrecer — devolver la lista vacía en ese punto es la
respuesta honesta, no un fallo a corregir con un tercer intento.

Quién la usa: el endpoint `/chat` en `app/routers/frontend.py`, envolviendo su llamada a
`top_k_gold_evolution`. Ver `.tmp/tasks2.md` tarea 2 y `.tmp/advanced_techniques.md` §5."""

from collections.abc import Awaitable, Callable
from typing import TypeVar

T = TypeVar("T")


async def retrieve_with_correction(
    retrieve: Callable[[float | None], Awaitable[list[T]]],
    max_distance: float | None,
    relaxed_max_distance: float | None = None,
) -> list[T]:
    """Calls `retrieve(max_distance)` first. If that returns an empty list, calls
    `retrieve(relaxed_max_distance)` once and returns whatever that finds (which may itself be
    empty — that is a genuine "nothing relevant", not an error). If the first call already
    finds something, `relaxed_max_distance` is never even evaluated: a strict, successful
    search is never second-guessed by a laxer one."""
    rows = await retrieve(max_distance)
    if rows:
        return rows
    return await retrieve(relaxed_max_distance)
