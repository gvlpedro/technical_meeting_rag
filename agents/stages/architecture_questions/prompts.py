"""Prompt construction for the architecture-questions stage. Each LLM call this stage makes
has one `.jinja` file, one path constant, one loader, and one builder function.
`build_architecture_question_generation_prompt` is production's own single combined call.
`build_architecture_identification_prompt`, `_drafting_prompt`, and `_selection_prompt` are
the decomposed-pipeline evaluation prototype. See
`agents.stages.architecture_questions.schemas`'s own docstring."""

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
    """`prompts/architecture_questions/template.md`'s raw text. This is the
    `ARCHITECTURE_CHANGES` requirements spec that this stage's prompts draft their
    clarification questions against. It covers everything except the full data-contract
    specs. See `agents.stages.data_contract_questions.prompts.load_data_contract_template`
    for that."""
    return _TEMPLATE_PATH.read_text(encoding="utf-8")


def load_architecture_question_generation_role() -> str:
    """`prompts/architecture_questions/combined.jinja`'s raw text. Its
    `{{architecture_changes}}`, `{{transcript}}`, and `{{known_architecture}}` placeholders
    are still unfilled. This is the production graph's single combined call for this stage."""
    return _COMBINED_ROLE_PATH.read_text(encoding="utf-8")


def load_architecture_identification_role() -> str:
    """`prompts/architecture_questions/identification.jinja`'s raw text. Its `{{transcript}}`
    and `{{known_architecture}}` placeholders are still unfilled. This is call 1 of the
    decomposed architecture-questions pipeline. It only identifies things; it does not draft
    questions."""
    return _IDENTIFICATION_ROLE_PATH.read_text(encoding="utf-8")


def load_architecture_question_drafting_role() -> str:
    """`prompts/architecture_questions/drafting.jinja`'s raw text. Its
    `{{architecture_changes}}`, `{{transcript}}`, `{{known_architecture}}`,
    `{{identified_components}}`, and `{{identified_data_contracts}}` placeholders are still
    unfilled. This is call 2 of the decomposed architecture-questions pipeline. It builds a
    broad, unfiltered candidate list."""
    return _DRAFTING_ROLE_PATH.read_text(encoding="utf-8")


def load_architecture_question_selection_role() -> str:
    """`prompts/architecture_questions/selection.jinja`'s raw text. Its `{{transcript}}` and
    `{{candidate_questions}}` placeholders are still unfilled. This is call 3 of the
    decomposed architecture-questions pipeline. It selects, deduplicates, and scores the
    candidates by value, down to the final set."""
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
    """Renders `prompts/architecture_questions/combined.jinja` as a real Jinja template. The
    prompt lives in that file, not duplicated here. So there is exactly one place to edit it,
    with no risk of the two copies drifting apart. This is question-generation stage 1 of 2:
    ADR, components, and data-contract identification. See `agents.stages.
    data_contract_questions.prompts.build_data_contract_question_generation_prompt` for stage
    2, the full ODCS-completeness questions. `architecture_diagram` is the Mermaid diagram of
    the architecture already known as of this transcript, if any. It is empty when none exists
    yet, for example on a first-time architecture description. The prompt's own
    `{% if known_architecture %}` handles that case; we do not special-case it here. The Jinja
    variable names (`architecture_changes`, `known_architecture`, `transcript`) belong to the
    prompt file itself. This function's own parameter names stay stable for callers, even if
    the prompt's internal naming changes."""
    role_template = jinja2.Template(load_architecture_question_generation_role())
    prompt = role_template.render(
        architecture_changes=template_text,
        transcript=transcript_text,
        known_architecture=architecture_diagram,
    )
    return [{"role": "user", "content": prompt}]


# --- Decomposed architecture-questions pipeline. This is an evaluation prototype. See
# agents.stages.architecture_questions.schemas.ArchitectureIdentificationResult's own docstring
# for why this exists alongside build_architecture_question_generation_prompt above, not
# instead of it. ----------------------------------------------------------------------------


def build_architecture_identification_prompt(transcript_text: str, architecture_diagram: str = "") -> list[dict]:
    """Renders `prompts/architecture_questions/identification.jinja`. This is call 1 of 3: it
    identifies components and data contracts only, and does not draft questions.
    `architecture_diagram` is the known-architecture Mermaid diagram. It has the same meaning,
    and the same empty-when-none-yet handling, as
    `build_architecture_question_generation_prompt`'s own parameter of the same name."""
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
    """Renders `prompts/architecture_questions/drafting.jinja`. This is call 2 of 3: it builds
    a broad, unfiltered candidate question list from call 1's identification result
    (`mentioned_components` and `mentioned_data_contracts`, already resolved by
    `build_architecture_identification_prompt`), plus the transcript and requirements spec."""
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
    """The classification stage's own `_generated_questions_block` only needs the id and
    question text, which is enough for classification. Selection is different: it needs every
    field. `scope`, `target`, and `requirement` are what let the LLM judge duplication and
    redundancy across candidates that share a target but do not have identical wording."""
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
    """Renders `prompts/architecture_questions/selection.jinja`. This is call 3 of 3: it takes
    call 2's broad candidate list and filters, deduplicates, and value-scores it down to the
    final set."""
    role_template = jinja2.Template(load_architecture_question_selection_role())
    prompt = role_template.render(
        transcript=transcript_text, candidate_questions=_full_questions_block(candidate_questions)
    )
    return [{"role": "user", "content": prompt}]
