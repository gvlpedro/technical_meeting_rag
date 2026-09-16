import io
import re
from datetime import date, datetime
from pathlib import Path

import tiktoken
from pypdf import PdfReader

_TAG_RE = re.compile(r"<[^>]+>")
_ENCODING = tiktoken.get_encoding("cl100k_base")


def parse_ingestion_date(value: str) -> date:
    """Parse a compact `YYYYMMDD` ingestion date, e.g. `20260906` (from `?ingestion_date=`
    or an `ingestion_date=<value>` folder name). Raises `ValueError` if malformed."""
    return datetime.strptime(value, "%Y%m%d").date()


def parse_vtt_text(raw: str) -> str:
    """Extract the plain-text transcript from raw WebVTT content, dropping cue
    timings/metadata. Split out from `parse_vtt` so an uploaded file's bytes (frontend upload
    tab) can go through the exact same parsing as a file already on disk, without a temp file
    in between."""
    lines = []
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped or stripped == "WEBVTT" or "-->" in stripped:
            continue
        if stripped.startswith(("Kind:", "Language:")):
            continue
        lines.append(_TAG_RE.sub("", stripped))
    return " ".join(lines)


def parse_vtt(path: Path) -> str:
    """Extract the plain-text transcript from a WebVTT file on disk — see `parse_vtt_text`
    for the actual parsing."""
    return parse_vtt_text(path.read_text(encoding="utf-8"))


def extract_pdf_text(content: bytes) -> str:
    """Extract plain text from an uploaded PDF's raw bytes, page by page, joined with a
    single space — same flat, unstructured shape `parse_vtt_text` produces for a transcript,
    so both feed `chunk_text` identically regardless of which format a meeting was uploaded
    as. No layout/table extraction, just `pypdf`'s own per-page `extract_text()`."""
    reader = PdfReader(io.BytesIO(content))
    return " ".join(page.extract_text() or "" for page in reader.pages)


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
