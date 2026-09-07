import re
from datetime import date, datetime
from pathlib import Path

import tiktoken

_TAG_RE = re.compile(r"<[^>]+>")
_ENCODING = tiktoken.get_encoding("cl100k_base")


def parse_ingestion_date(value: str) -> date:
    """Parse a compact `YYYYMMDD` ingestion date, e.g. `20260906` (from `?ingestion_date=`
    or an `ingestion_date=<value>` folder name). Raises `ValueError` if malformed."""
    return datetime.strptime(value, "%Y%m%d").date()


def parse_vtt(path: Path) -> str:
    """Extract the plain-text transcript from a WebVTT file, dropping cue timings/metadata."""
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped == "WEBVTT" or "-->" in stripped:
            continue
        if stripped.startswith(("Kind:", "Language:")):
            continue
        lines.append(_TAG_RE.sub("", stripped))
    return " ".join(lines)


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
