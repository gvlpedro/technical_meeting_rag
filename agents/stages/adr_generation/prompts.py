"""This file builds the prompt for the ADR-generation stage. This stage is the pipeline's Actor.
It makes one LLM call per source. That call writes the final ADR directly from a transcript plus
its resolved clarifications. There is no intermediate structured JSON step (`doc/silver_process.md`
§3 node 5)."""

from typing import TypedDict

import jinja2

from agents.template import PROMPTS_DIR

_ROLE_PATH = PROMPTS_DIR / "adr_generation" / "generator.jinja"


def load_adr_generation_role() -> str:
    """Returns the raw text of `prompts/adr_generation/generator.jinja`. The `{{transcript}}`
    and `{{clarifications}}` placeholders are still empty here. `build_adr_generation_prompt`
    fills them in. The production graph's `synthesize_document` node uses this
    (`agents/graph.py`). The `agents/stages/adr_generation/testing/` golden set also tests this on its own."""
    return _ROLE_PATH.read_text(encoding="utf-8")


class QaPair(TypedDict):
    """One resolved clarification for `build_adr_generation_prompt`. This is only a
    question/answer pair. It is not the full `ClarificationItem` shape. `adr_generator.jinja`
    does not need `id`, `scope`, `target`, or `requirement`. It only needs the text a human would
    read. `answer` is `None` when the question was asked but never got an answer. See
    `_qa_pairs_block`: it shows this case as `(not answered)`. It does not drop the question
    silently."""

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


def build_adr_generation_prompt(
    transcript_text: str, clarifications: list[QaPair], previous_architecture_diagram: str = ""
) -> list[dict]:
    """Renders `prompts/adr_generation/generator.jinja` as a real Jinja template. See that file
    for the actual generation rules: component-inclusion discipline, no placeholders, and
    `doc/adr_example.md`-shaped output. This function takes a flat question/answer list. It does
    not take the graph's own `ClarificationItem` state shape. See `QaPair` for that list's shape.

    `previous_architecture_diagram` is Gold's own current, tenant-wide architecture state. See
    `agents.graph.synthesize_document`: every source in a batch gets this same diagram, because
    Gold is reconciled across all sources, not scoped to any single one. It is empty only when
    Gold has no live component for this tenant yet. This diagram grounds §2 "Previous
    Architecture" in what was actually last published. Without it, the model would have to
    reconstruct that section (or skip it) purely from this run's own transcript and
    clarifications.

    The "regenerate with feedback" endpoint in `app/routers/frontend.py` passes a different value
    here instead: `agents.stages.adr_generation.service.own_previous_architecture_diagram`. That
    value is this very draft's own §2, unchanged by the new feedback. It is not Gold's state
    again."""
    role_template = jinja2.Template(load_adr_generation_role())
    prompt = role_template.render(
        transcript=transcript_text,
        clarifications=_qa_pairs_block(clarifications),
        previous_architecture_diagram=previous_architecture_diagram,
    )
    return [{"role": "user", "content": prompt}]
