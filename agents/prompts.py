from typing import TypedDict

import jinja2

from agents.state import (
    ClarificationItem,
    GeneratedQuestion,
    MentionedComponentItem,
    MentionedDataContractItem,
)
from agents.template import (
    load_adr_critic_role,
    load_adr_generation_role,
    load_architecture_question_generation_role,
    load_data_contract_question_generation_role,
    load_question_classifier_role,
)


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


def _mentioned_data_contracts_block(contracts: list[MentionedDataContractItem]) -> str:
    if not contracts:
        return "(none identified yet)"
    return "\n".join(f"- {c['name']}: {c['producer']} -> {c['consumer']} ({c['action']})" for c in contracts)


def build_architecture_question_generation_prompt(
    template_text: str, transcript_text: str, architecture_diagram: str = ""
) -> list[dict]:
    """Renders `prompts/architecture_questions.jinja` as a real Jinja
    template — the prompt lives in that file, not duplicated here, so there's exactly
    one place to edit it and no drift risk. Question-generation stage 1 of 2: ADR,
    components, and data-contract *identification* (see `build_data_contract_question_
    generation_prompt` for stage 2, full ODCS-completeness questions). `architecture_diagram`
    is the Mermaid diagram of the architecture already known as of this transcript, if any —
    empty when none exists yet (e.g. a first-time architecture description); the prompt's own
    `{% if known_architecture %}` handles that case, it isn't special-cased here. The Jinja
    variable names (`architecture_changes`/`known_architecture`/`transcript`) are the prompt
    file's own — this function's own parameter names stay stable for callers even if the
    prompt's internal naming changes."""
    role_template = jinja2.Template(load_architecture_question_generation_role())
    prompt = role_template.render(
        architecture_changes=template_text,
        transcript=transcript_text,
        known_architecture=architecture_diagram,
    )
    return [{"role": "user", "content": prompt}]


def build_data_contract_question_generation_prompt(
    template_text: str, transcript_text: str, mentioned_data_contracts: list[MentionedDataContractItem]
) -> list[dict]:
    """Renders `prompts/data_contract_questions.jinja`. Question-generation
    stage 2 of 2: full ODCS-completeness questions for every contract stage 1
    (`build_architecture_question_generation_prompt`) already identified — this stage never
    discovers a contract on its own, only drafts questions for the fixed list it's given."""
    role_template = jinja2.Template(load_data_contract_question_generation_role())
    prompt = role_template.render(
        data_contract_requirements=template_text,
        transcript=transcript_text,
        identified_data_contracts=_mentioned_data_contracts_block(mentioned_data_contracts),
    )
    return [{"role": "user", "content": prompt}]


def _generated_questions_block(questions: list[GeneratedQuestion]) -> str:
    return "\n".join(f"- id: {q['id']}\n  question: {q['question']}" for q in questions)


def build_classification_prompt(questions: list[GeneratedQuestion], transcript_text: str) -> list[dict]:
    """Renders `prompts/question_classifier.jinja` as a real Jinja template — same
    single-user-message convention as every other prompt builder in this module, not a
    separate system+user pair (that split lived only in this function's own Python string
    before; the instructions are now the template's own opening section instead)."""
    role_template = jinja2.Template(load_question_classifier_role())
    prompt = role_template.render(
        questions=_generated_questions_block(questions),
        transcript=transcript_text,
    )
    return [{"role": "user", "content": prompt}]


def _qa_pairs_block(clarifications: list[QaPair]) -> str:
    if not clarifications:
        return "(no clarifications were collected for this transcript)"
    lines = []
    for pair in clarifications:
        answer = pair["answer"] if pair["answer"] is not None else "(not answered)"
        lines.append(f"- Q: {pair['question']}\n  A: {answer}")
    return "\n".join(lines)


def build_adr_generation_prompt(transcript_text: str, clarifications: list[QaPair]) -> list[dict]:
    """Renders `prompts/adr_generator.jinja` as a real Jinja template — see
    that file for the actual generation rules (component-inclusion discipline, no
    placeholders, `doc/adr_example.md`-shaped output). Takes a flat question/answer list
    rather than the graph's own `ClarificationItem` state shape — see `QaPair`."""
    role_template = jinja2.Template(load_adr_generation_role())
    prompt = role_template.render(
        transcript=transcript_text,
        clarifications=_qa_pairs_block(clarifications),
    )
    return [{"role": "user", "content": prompt}]


def build_critic_prompt(
    document: str, source_content: str, clarifications: list[ClarificationItem]
) -> list[dict]:
    """Renders `prompts/adr_critic.jinja` as a real Jinja template — same
    single-user-message convention as every other prompt builder in this module."""
    role_template = jinja2.Template(load_adr_critic_role())
    prompt = role_template.render(
        document=document,
        clarifications=_clarifications_block(clarifications),
        source_content=source_content,
    )
    return [{"role": "user", "content": prompt}]
