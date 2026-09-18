from functools import lru_cache
from typing import TYPE_CHECKING

from app.config import settings

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=1)
def _get_model() -> "SentenceTransformer":
    from sentence_transformers import SentenceTransformer  # This import is heavy (it pulls in torch), so we delay it until first use.

    return SentenceTransformer(settings.embedding_model)


def embed(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts. This call blocks. From async code, call it through
    `asyncio.to_thread`."""
    model = _get_model()
    return model.encode(texts, convert_to_numpy=True).tolist()
