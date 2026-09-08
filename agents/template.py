import json
import re
from pathlib import Path

_TEMPLATE_PATH = Path(__file__).resolve().parent.parent / "doc" / "clarification_template.md"
_QUESTION_GENERATION_ROLE_PATH = (
    Path(__file__).resolve().parent.parent
    / "prompting"
    / "roles"
    / "common"
    / "clarification_questions.jinja"
)

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def load_template() -> str:
    """`doc/clarification_template.md`'s raw text — the exact shape `synthesize_document`
    fills in, per `doc/silver_process.md` §1."""
    return _TEMPLATE_PATH.read_text(encoding="utf-8")


def load_question_generation_role() -> str:
    """`prompting/roles/common/clarification_questions.jinja`'s raw text, `{{template}}`/
    `{{transcript}}` placeholders still unfilled — `generate_questions` fills them."""
    return _QUESTION_GENERATION_ROLE_PATH.read_text(encoding="utf-8")


def extract_json(content: str) -> str:
    """Strip a ```json fence if the model wrapped its structured output in one,
    despite being asked not to. A no-op on already-bare JSON."""
    return _FENCE_RE.sub("", content).strip()


def load_json_response(content: str) -> dict:
    return json.loads(extract_json(content))
