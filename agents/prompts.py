from typing import TypedDict

import jinja2

from agents.state import ClarificationItem, GeneratedQuestion, GoldComponentSnapshot, MentionedComponentItem
from agents.template import load_adr_generation_role, load_question_generation_role


class QaPair(TypedDict):
    """One resolved clarification for `build_adr_generation_prompt` — deliberately just a
    question/answer pair, not the full `ClarificationItem` shape: `adr_generator.jinja`
    doesn't need `id`/`scope`/`target`/`requirement`, only the text a human would read.
    `answer` is `None` when the question was asked but never answered — see
    `_qa_pairs_block`, which renders that as `(not answered)`, not silently dropped."""

    question: str
    answer: str | None


def _clarifications_block(clarifications: list[ClarificationItem]) -> str:
    if not clarifications:
        return "(none yet)"
    lines = []
    for item in clarifications:
        answer = item["answer"] if item["answer"] is not None else "(unknown)"
        lines.append(f"- [{item['target']}] Q: {item['question']}\n  A: {answer}")
    return "\n".join(lines)


def _mentioned_components_block(components: list[MentionedComponentItem]) -> str:
    if not components:
        return "(none identified yet)"
    return "\n".join(f"- {c['name']}: {c['status']}" for c in components)


def _gold_components_block(components: list[GoldComponentSnapshot]) -> str:
    if not components:
        return "(none known yet)"
    return "\n".join(f"- {c['name']}: {c.get('description', '')}" for c in components)


def build_question_generation_prompt(
    template_text: str, transcript_text: str, architecture_diagram: str = ""
) -> list[dict]:
    """Renders `prompting/roles/common/clarification_questions.jinja` as a real Jinja
    template — the prompt lives in that file, not duplicated here, so there's exactly
    one place to edit it and no drift risk. `architecture_diagram` is the Mermaid
    diagram of the architecture already known as of this transcript, if any — empty
    when none exists yet (e.g. a first-time architecture description); the prompt's
    own `{% if known_architecture %}` handles that case, it isn't special-cased here.
    The Jinja variable names (`architecture_changes`/`known_architecture`/`transcript`)
    are the prompt file's own — this function's own parameter names stay stable for
    callers even if the prompt's internal naming changes."""
    role_template = jinja2.Template(load_question_generation_role())
    prompt = role_template.render(
        architecture_changes=template_text,
        transcript=transcript_text,
        known_architecture=architecture_diagram,
    )
    return [{"role": "user", "content": prompt}]


def build_classification_prompt(questions: list[GeneratedQuestion], transcript_text: str) -> list[dict]:
    system = (
        "You classify each question below against a meeting transcript. For every question, "
        "decide exactly one outcome:\n"
        '- "answered": the transcript states it, or implies it unambiguously. Include the answer.\n'
        '- "unknown": the transcript does not say, and it plausibly never will be known from this '
        "meeting alone. Leave answer null.\n"
        '- "needs_clarification": the transcript does not say, but a human involved in the meeting '
        "would plausibly know. Leave answer null.\n"
        "Return every question exactly once, echoing back its own `id` unchanged, as JSON matching "
        "the requested schema — no commentary outside the JSON."
    )
    questions_block = "\n".join(f"- id: {q['id']}\n  question: {q['question']}" for q in questions)
    user = f"Questions:\n{questions_block}\n\nTranscript:\n{transcript_text}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_synthesis_prompt(
    template_text: str,
    source_content: str,
    clarifications: list[ClarificationItem],
    known_gold_components: list[GoldComponentSnapshot],
    mentioned_components: list[MentionedComponentItem],
) -> list[dict]:
    system = (
        "You fill in the architecture description template below directly, in Markdown, using "
        "only the transcript and clarifications given for this one source. Fill every section the "
        "clarifications actually support: Motivation and Context, Affected Components (tag each "
        "`new`, `modified`, `removed`, `unchanged`, or `unknown` — start from the status already "
        "determined for it below, only overriding it if a clarification resolved an `unknown`, with "
        "a one-line reason each), Open Data Contracts as ODCS v3.0.0 JSON drafts (set `evidenced: "
        "false` and `confirmed: false` on any field the transcript did not literally name), "
        "Alternatives, Consequences, Risks, and so on down what the clarifications cover. Leave "
        "every section nobody addressed — including the architecture diagram, the Decision "
        "Criteria table, Validation metrics, and Migration/Rollback strategy — exactly as the "
        "template's own placeholder text. Never invent content for a section with no evidence; a "
        "partially filled template is the correct, expected output. Output the complete filled "
        "template and nothing else — no commentary before or after it."
    )
    user = (
        f"Template:\n{template_text}\n\n"
        f"Components mentioned in this batch and their lifecycle status (as already determined "
        f"when the clarification questions were drafted):\n"
        f"{_mentioned_components_block(mentioned_components)}\n\n"
        f"Known existing components (Gold, best-effort — may be empty):\n"
        f"{_gold_components_block(known_gold_components)}\n\n"
        f"Clarifications (pooled across the whole ingestion batch):\n"
        f"{_clarifications_block(clarifications)}\n\n"
        f"Transcript for this source only:\n{source_content}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _qa_pairs_block(clarifications: list[QaPair]) -> str:
    if not clarifications:
        return "(no clarifications were collected for this transcript)"
    lines = []
    for pair in clarifications:
        answer = pair["answer"] if pair["answer"] is not None else "(not answered)"
        lines.append(f"- Q: {pair['question']}\n  A: {answer}")
    return "\n".join(lines)


def build_adr_generation_prompt(transcript_text: str, clarifications: list[QaPair]) -> list[dict]:
    """Renders `prompting/roles/common/adr_generator.jinja` as a real Jinja template — see
    that file for the actual generation rules (component-inclusion discipline, no
    placeholders, `doc/adr_example.md`-shaped output). Distinct from `build_synthesis_prompt`:
    that one fills the full `prompting/roles/common/clarification_template.md` template inline; this one always
    produces the shorter, ADR-only shape `doc/adr_example.md` demonstrates, from a flat
    question/answer list rather than the graph's own `ClarificationItem` state shape."""
    role_template = jinja2.Template(load_adr_generation_role())
    prompt = role_template.render(
        transcript=transcript_text,
        clarifications=_qa_pairs_block(clarifications),
    )
    return [{"role": "user", "content": prompt}]


def build_critic_prompt(
    document: str, source_content: str, clarifications: list[ClarificationItem]
) -> list[dict]:
    system = (
        "You review a drafted architecture document against its source transcript and "
        "clarifications. For every substantive claim in the document (a component's lifecycle tag, "
        "a data-contract field, an overview statement), decide whether it is actually supported. "
        "Quote `claim` as a verbatim, exact substring copied from the document text — never "
        'paraphrased — so it can be located later. Mark `severity` as "material" only when the '
        "claim actively contradicts the transcript or clarifications, not merely when it lacks "
        'evidence (an honestly `evidenced: false` field is expected, not a defect — mark those '
        '"low" if you flag them at all). Only flag claims that are actually wrong or unsupported; '
        "do not flag every sentence. Return JSON matching the requested schema, nothing else."
    )
    user = (
        f"Document:\n{document}\n\n"
        f"Clarifications:\n{_clarifications_block(clarifications)}\n\n"
        f"Transcript for this source only:\n{source_content}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
