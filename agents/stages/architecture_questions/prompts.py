"""Prompt construction for the architecture-questions stage.
`build_architecture_question_generation_prompt` is production's combined call. The identification,
drafting, and selection builders are the decomposed-pipeline prototype — see
`agents.stages.architecture_questions.schemas`."""

import jinja2

from agents.state import GeneratedQuestion, MentionedComponentItem, MentionedDataContractItem
from agents.template import PROMPTS_DIR

_STAGE_DIR = PROMPTS_DIR / "architecture_questions"
_COMBINED_ROLE_PATH = _STAGE_DIR / "combined.jinja"
_IDENTIFICATION_ROLE_PATH = _STAGE_DIR / "identification.jinja"
_DRAFTING_ROLE_PATH = _STAGE_DIR / "drafting.jinja"
_SELECTION_ROLE_PATH = _STAGE_DIR / "selection.jinja"
_TEMPLATE_PATH = _STAGE_DIR / "template.md"


def load_architecture_template() -> str:
    """`template.md`'s raw text: the `ARCHITECTURE_CHANGES` requirements spec. Covers
    everything except full data-contract specs (see `data_contract_questions.prompts`)."""
    return _TEMPLATE_PATH.read_text(encoding="utf-8")


def load_architecture_question_generation_role() -> str:
    """`combined.jinja`'s raw text, placeholders unfilled. Production's single call for this
    stage."""
    return _COMBINED_ROLE_PATH.read_text(encoding="utf-8")


def load_architecture_identification_role() -> str:
    """`identification.jinja`'s raw text, placeholders unfilled. Call 1 of 3: identifies
    things, drafts no questions."""
    return _IDENTIFICATION_ROLE_PATH.read_text(encoding="utf-8")


def load_architecture_question_drafting_role() -> str:
    """`drafting.jinja`'s raw text, placeholders unfilled. Call 2 of 3: builds a broad,
    unfiltered candidate list."""
    return _DRAFTING_ROLE_PATH.read_text(encoding="utf-8")


def load_architecture_question_selection_role() -> str:
    """`selection.jinja`'s raw text, placeholders unfilled. Call 3 of 3: selects,
    deduplicates, and value-scores down to the final set."""
    return _SELECTION_ROLE_PATH.read_text(encoding="utf-8")


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
    """Renders `combined.jinja`: stage 1 of 2, ADR/components/data-contract identification.
    Stage 2 (full ODCS questions) is `data_contract_questions.prompts`.
    `architecture_diagram` is the known Mermaid diagram, empty if none exists yet — the
    template's own `{% if known_architecture %}` handles that case."""
    role_template = jinja2.Template(load_architecture_question_generation_role())
    prompt = role_template.render(
        architecture_changes=template_text,
        transcript=transcript_text,
        known_architecture=architecture_diagram,
    )
    return [{"role": "user", "content": prompt}]


# --- Decomposed architecture-questions pipeline (evaluation prototype). See
# architecture_questions.schemas.ArchitectureIdentificationResult for why this exists
# alongside build_architecture_question_generation_prompt above, not instead of it. -----------


def build_architecture_identification_prompt(transcript_text: str, architecture_diagram: str = "") -> list[dict]:
    """Renders `identification.jinja`: call 1 of 3, identifies components and data contracts
    only. `architecture_diagram` works the same as in
    `build_architecture_question_generation_prompt`."""
    role_template = jinja2.Template(load_architecture_identification_role())
    prompt = role_template.render(transcript=transcript_text, known_architecture=architecture_diagram)
    return [{"role": "user", "content": prompt}]


def build_architecture_question_drafting_prompt(
    template_text: str,
    transcript_text: str,
    architecture_diagram: str,
    mentioned_components: list[MentionedComponentItem],
    mentioned_data_contracts: list[MentionedDataContractItem],
) -> list[dict]:
    """Renders `drafting.jinja`: call 2 of 3, builds a broad, unfiltered candidate list from
    call 1's result plus the transcript and requirements spec."""
    role_template = jinja2.Template(load_architecture_question_drafting_role())
    prompt = role_template.render(
        architecture_changes=template_text,
        transcript=transcript_text,
        known_architecture=architecture_diagram,
        identified_components=_mentioned_components_block(mentioned_components),
        identified_data_contracts=_mentioned_data_contracts_block(mentioned_data_contracts),
    )
    return [{"role": "user", "content": prompt}]


def _full_questions_block(questions: list[GeneratedQuestion]) -> str:
    """Classification's own block only needs id and question text. Selection needs every
    field — `scope`/`target`/`requirement` let the LLM judge duplicates with different
    wording."""
    if not questions:
        return "(no candidates drafted)"
    return "\n".join(
        f"- id: {q['id']}\n  scope: {q['scope']}\n  target: {q['target']}\n"
        f"  requirement: {q['requirement']}\n  question: {q['question']}"
        for q in questions
    )


def build_architecture_question_selection_prompt(
    transcript_text: str, candidate_questions: list[GeneratedQuestion]
) -> list[dict]:
    """Renders `selection.jinja`: call 3 of 3, filters and value-scores call 2's candidates
    down to the final set."""
    role_template = jinja2.Template(load_architecture_question_selection_role())
    prompt = role_template.render(
        transcript=transcript_text, candidate_questions=_full_questions_block(candidate_questions)
    )
    return [{"role": "user", "content": prompt}]
