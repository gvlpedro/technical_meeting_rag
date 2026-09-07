from functools import lru_cache
from typing import TYPE_CHECKING

from app.config import settings

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=1)
def _get_model() -> "SentenceTransformer":
    from sentence_transformers import SentenceTransformer  # heavy import (torch); deferred to first use

    return SentenceTransformer(settings.embedding_model)


def embed(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts, blocking. Call via `asyncio.to_thread` from async code."""
    model = _get_model()
    return model.encode(texts, convert_to_numpy=True).tolist()
