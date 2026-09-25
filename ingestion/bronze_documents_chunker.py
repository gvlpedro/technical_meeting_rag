from datetime import date, datetime

import tiktoken

_ENCODING = tiktoken.get_encoding("cl100k_base")


def parse_ingestion_date(value: str) -> date:
    """Parse a compact `YYYYMMDD` ingestion date, for example `20260906`. This value comes
    from `?ingestion_date=` or from an `ingestion_date=<value>` folder name. Raises
    `ValueError` if the value is not well formed."""
    return datetime.strptime(value, "%Y%m%d").date()


def chunk_text(text: str, chunk_size: int = 400, chunk_overlap: int = 50) -> list[str]:
    """Split `text` into token-bounded, overlapping chunks."""
    tokens = _ENCODING.encode(text)
    if not tokens:
        return []

    chunks = []
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunks.append(_ENCODING.decode(tokens[start:end]))
        if end >= len(tokens):
            break
        start = end - chunk_overlap
    return chunks
