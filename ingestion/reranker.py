from functools import lru_cache
from typing import TYPE_CHECKING

from app.config import settings

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder


@lru_cache(maxsize=1)
def _get_model() -> "CrossEncoder":
    from sentence_transformers import CrossEncoder  # This import is heavy (it pulls in torch), so we delay it until first use.

    return CrossEncoder(settings.reranker_model)


def score_candidates(query: str, documents: list[str]) -> list[float]:
    """Scores each of `documents` against `query` with a local cross-encoder — one score per
    document, higher means more relevant, same order as `documents`. Unlike `embedder.embed`,
    a cross-encoder reads the (query, document) pair TOGETHER in one forward pass, instead of
    embedding each side separately and comparing the two vectors afterward. That joint reading
    is what lets it catch relevance a cosine-distance ranking can miss — and also why it costs
    one model call per candidate, not one shared embedding lookup, so it is only worth paying
    for a modest number of candidates, never a whole corpus.

    This call blocks. From async code, call it through `asyncio.to_thread`, the same way
    `embedder.embed` already is."""
    model = _get_model()
    return model.predict([(query, document) for document in documents]).tolist()
