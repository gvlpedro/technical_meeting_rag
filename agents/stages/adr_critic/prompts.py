"""This file builds the prompt for the ADR-critic stage. The stage has one `.jinja` file, one
path constant, one loader function, and one builder function."""

import jinja2

from agents.state import ClarificationItem
from agents.template import PROMPTS_DIR

_ROLE_PATH = PROMPTS_DIR / "adr_critic" / "critic.jinja"


def load_adr_critic_role() -> str:
    """Returns the raw text of `prompts/adr_critic/critic.jinja`. The `{{document}}`,
    `{{clarifications}}`, and `{{source_content}}` placeholders are still empty here.
    `build_critic_prompt` fills them in. The production graph's `critic_document` node uses this
    (`agents/graph.py`)."""
    return _ROLE_PATH.read_text(encoding="utf-8")


def _clarifications_block(clarifications: list[ClarificationItem]) -> str:
    if not clarifications:
        return "(none yet)"
    lines = []
    for item in clarifications:
        answer = item["answer"] if item["answer"] is not None else "(unknown)"
        lines.append(f"- [{item['target']}] Q: {item['question']}\n  A: {answer}")
    return "\n".join(lines)


def build_critic_prompt(
    document: str, source_content: str, clarifications: list[ClarificationItem]
) -> list[dict]:
    """Renders `prompts/adr_critic/critic.jinja` as a real Jinja template. This follows the same
    single-user-message convention every prompt builder in this codebase uses."""
    role_template = jinja2.Template(load_adr_critic_role())
    prompt = role_template.render(
        document=document,
        clarifications=_clarifications_block(clarifications),
        source_content=source_content,
    )
    return [{"role": "user", "content": prompt}]
