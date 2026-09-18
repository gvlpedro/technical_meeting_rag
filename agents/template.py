"""Generic prompt and response tools shared by every stage. This file holds the `prompts/`
directory's base path. It also holds JSON extraction for structured-output responses.
Sometimes a response arrives wrapped in a ```json fence, even though we ask the model not to
do that. Each stage keeps its own `.jinja` path constants and `load_*_role()` file readers in
that stage's own `agents/stages/<stage>/prompts.py`, not here. We keep them next to the
prompt-builder function that uses them. That way, opening one stage's `prompts.py` shows the
whole chain in one place: which file, which loader, which builder."""

import json
import re
from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def extract_json(content: str) -> str:
    """Remove a ```json fence if the model wrapped its structured output in one,
    even though we ask it not to. Does nothing if the JSON has no fence already."""
    return _FENCE_RE.sub("", content).strip()


def load_json_response(content: str) -> dict:
    return json.loads(extract_json(content))
