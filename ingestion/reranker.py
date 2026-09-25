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
    document, higher means more relevant, same order as `documents`."""
    model = _get_model()
    return model.predict([(query, document) for document in documents]).tolist()
