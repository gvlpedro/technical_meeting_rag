"""TECHNIQUE: Corrective RAG (one retry with a looser filter).

Problem: a strict `max_distance` filter avoids answering from a weak match when nothing is
relevant. But it can also hide a real fact whose distance landed just past the cutoff.

How: `retrieve_with_correction` takes a `retrieve` function and calls it with the strict
filter first. If that returns nothing, it calls `retrieve` once more with a looser filter. It
knows nothing about Gold, Postgres, or embeddings, so a test can pass it a fake function.

One retry only. If the second call also finds nothing, the question has no relevant fact. An
empty list at that point is the honest answer, not a failure to fix with a third try.

Used by: the `/chat` endpoint in `app/routers/frontend.py`, around `top_k_gold_evolution`."""

from collections.abc import Awaitable, Callable
from typing import TypeVar

T = TypeVar("T")


async def retrieve_with_correction(
    retrieve: Callable[[float | None], Awaitable[list[T]]],
    max_distance: float | None,
    relaxed_max_distance: float | None = None,
) -> list[T]:
    """Calls `retrieve(max_distance)` first. If that list is empty, calls
    `retrieve(relaxed_max_distance)` once and returns its result, even if also empty. Never
    retries a search that already found something."""
    rows = await retrieve(max_distance)
    if rows:
        return rows
    return await retrieve(relaxed_max_distance)
