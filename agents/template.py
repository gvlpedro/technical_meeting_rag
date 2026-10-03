"""Shared prompt and response tools. Holds the `prompts/` base path and JSON extraction.
Each stage keeps its own `.jinja` paths and loaders in its own `prompts.py`."""

import json
import re
from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def extract_json(content: str) -> str:
    """Remove a ```json fence around the content, if present."""
    return _FENCE_RE.sub("", content).strip()


def load_json_response(content: str) -> dict:
    return json.loads(extract_json(content))
