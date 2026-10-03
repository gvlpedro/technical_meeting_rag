"""Builds the prompt for the ADR-generation stage (the pipeline's Actor). One LLM call per
source writes the final ADR from a transcript plus its clarifications — no intermediate JSON
step (`doc/silver_process.md` §3 node 5)."""

from typing import TypedDict

import jinja2

from agents.state import MentionedComponentItem, MentionedDataContractItem
from agents.template import PROMPTS_DIR

_ROLE_PATH = PROMPTS_DIR / "adr_generation" / "generator.jinja"


def load_adr_generation_role() -> str:
    """`generator.jinja`'s raw text, placeholders unfilled. `build_adr_generation_prompt` fills
    them in."""
    return _ROLE_PATH.read_text(encoding="utf-8")


class QaPair(TypedDict):
    """A question/answer pair, not the full `ClarificationItem` shape. `answer` is `None` when
    unanswered — `_qa_pairs_block` shows that as `(not answered)`, never dropped silently."""

    question: str
    answer: str | None


def _qa_pairs_block(clarifications: list[QaPair]) -> str:
    if not clarifications:
        return "(no clarifications were collected for this transcript)"
    lines = []
    for pair in clarifications:
        answer = pair["answer"] if pair["answer"] is not None else "(not answered)"
        lines.append(f"- Q: {pair['question']}\n  A: {answer}")
    return "\n".join(lines)


def _mentioned_components_block(components: list[MentionedComponentItem]) -> str:
    if not components:
        return "(none identified)"
    return "\n".join(f"- {c['name']}: {c['status']}" for c in components)


def _mentioned_data_contracts_block(contracts: list[MentionedDataContractItem]) -> str:
    if not contracts:
        return "(none identified)"
    return "\n".join(f"- {c['name']}: {c['producer']} -> {c['consumer']} ({c['action']})" for c in contracts)


def build_adr_generation_prompt(
    transcript_text: str,
    clarifications: list[QaPair],
    previous_architecture_diagram: str = "",
    mentioned_components: list[MentionedComponentItem] | None = None,
    mentioned_data_contracts: list[MentionedDataContractItem] | None = None,
) -> list[dict]:
    """Renders `generator.jinja`. See that file for the generation rules. Takes a flat
    question/answer list (`QaPair`), not the graph's `ClarificationItem` shape.

    `previous_architecture_diagram` is Gold's current, tenant-wide architecture state — every
    source in a batch gets the same one. Empty only when Gold has no live component yet. It
    grounds §2 "Previous Architecture" in what was last published. The "regenerate with
    feedback" endpoint instead passes `own_previous_architecture_diagram`: this draft's own
    §2, unchanged by the new feedback, not Gold's state again.

    `mentioned_components`/`mentioned_data_contracts` are the architecture-questions
    identification stage's own classification. Without them, this call could re-derive a
    stricter, inconsistent status — confirmed in a real case: excluding a component
    identification had already confirmed `new`. Passing them closes that gap: a confirmed
    entry here counts as `DOCUMENTED_CONTENT`, like an answered clarification. Both default to
    `None` for "regenerate with feedback", which has no access to this stage's output."""
    role_template = jinja2.Template(load_adr_generation_role())
    prompt = role_template.render(
        transcript=transcript_text,
        clarifications=_qa_pairs_block(clarifications),
        previous_architecture_diagram=previous_architecture_diagram,
        identified_components=_mentioned_components_block(mentioned_components or []),
        identified_data_contracts=_mentioned_data_contracts_block(mentioned_data_contracts or []),
    )
    return [{"role": "user", "content": prompt}]
