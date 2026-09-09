import json
import re
from pathlib import Path

_TEMPLATE_PATH = (
    Path(__file__).resolve().parent.parent / "prompting" / "roles" / "common" / "clarification_template.md"
)
_QUESTION_GENERATION_ROLE_PATH = (
    Path(__file__).resolve().parent.parent
    / "prompting"
    / "roles"
    / "common"
    / "clarification_questions.jinja"
)
_ADR_GENERATION_ROLE_PATH = (
    Path(__file__).resolve().parent.parent / "prompting" / "roles" / "common" / "adr_generator.jinja"
)

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def load_template() -> str:
    """`prompting/roles/common/clarification_template.md`'s raw text — the `ARCHITECTURE_CHANGES` requirements
    spec `generate_questions` drafts its clarification questions against (`clarification_
    questions.jinja`'s `{{ architecture_changes }}`). No longer what `synthesize_document`
    fills in — that node renders `adr_generator.jinja` instead, see `load_adr_generation_role`."""
    return _TEMPLATE_PATH.read_text(encoding="utf-8")


def load_question_generation_role() -> str:
    """`prompting/roles/common/clarification_questions.jinja`'s raw text, `{{template}}`/
    `{{transcript}}` placeholders still unfilled — `generate_questions` fills them."""
    return _QUESTION_GENERATION_ROLE_PATH.read_text(encoding="utf-8")


def load_adr_generation_role() -> str:
    """`prompting/roles/common/adr_generator.jinja`'s raw text, `{{transcript}}`/
    `{{clarifications}}` placeholders still unfilled — `build_adr_generation_prompt` fills
    them. Used by the production graph's `synthesize_document` node (`agents/graph.py`) and
    exercised in isolation by `testing_adr_acb/`'s golden set."""
    return _ADR_GENERATION_ROLE_PATH.read_text(encoding="utf-8")


def extract_json(content: str) -> str:
    """Strip a ```json fence if the model wrapped its structured output in one,
    despite being asked not to. A no-op on already-bare JSON."""
    return _FENCE_RE.sub("", content).strip()


def load_json_response(content: str) -> dict:
    return json.loads(extract_json(content))
