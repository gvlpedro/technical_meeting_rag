import json
import re
from pathlib import Path

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

# Flat — every prompt/template file hangs directly off prompts/, no roles/common/ subdirectory.
_ARCHITECTURE_TEMPLATE_PATH = _PROMPTS_DIR / "architecture_template.md"
_DATA_CONTRACT_TEMPLATE_PATH = _PROMPTS_DIR / "data_contract_template.md"
_ARCHITECTURE_QUESTION_ROLE_PATH = _PROMPTS_DIR / "architecture_questions.jinja"
_DATA_CONTRACT_QUESTION_ROLE_PATH = _PROMPTS_DIR / "data_contract_questions.jinja"
_ADR_GENERATION_ROLE_PATH = _PROMPTS_DIR / "adr_generator.jinja"
_QUESTION_CLASSIFIER_PATH = _PROMPTS_DIR / "question_classifier.jinja"
_ADR_CRITIC_PATH = _PROMPTS_DIR / "adr_critic.jinja"

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def load_architecture_template() -> str:
    """`prompts/architecture_template.md`'s raw text — the
    `ARCHITECTURE_CHANGES` requirements spec `generate_architecture_questions` drafts its
    clarification questions against (`architecture_questions.jinja`'s
    `{{ architecture_changes }}`). Covers everything except full data-contract specs — see
    `load_data_contract_template` for that."""
    return _ARCHITECTURE_TEMPLATE_PATH.read_text(encoding="utf-8")


def load_data_contract_template() -> str:
    """`prompts/data_contract_template.md`'s raw text — the per-contract ODCS
    completeness spec `generate_data_contract_questions` drafts its clarification questions
    against (`data_contract_questions.jinja`'s `{{ data_contract_requirements }}`), applied once
    per contract in `IDENTIFIED_DATA_CONTRACTS`."""
    return _DATA_CONTRACT_TEMPLATE_PATH.read_text(encoding="utf-8")


def load_architecture_question_generation_role() -> str:
    """`prompts/architecture_questions.jinja`'s raw text, `{{architecture_changes}}`/
    `{{transcript}}`/`{{known_architecture}}` placeholders still unfilled — question-generation
    stage 1 of 2 (ADR, components, data-contract *identification*). `generate_architecture_
    questions` fills them."""
    return _ARCHITECTURE_QUESTION_ROLE_PATH.read_text(encoding="utf-8")


def load_data_contract_question_generation_role() -> str:
    """`prompts/data_contract_questions.jinja`'s raw text,
    `{{data_contract_requirements}}`/`{{transcript}}`/`{{identified_data_contracts}}`
    placeholders still unfilled — question-generation stage 2 of 2 (full ODCS-completeness
    questions for each contract stage 1 identified). `generate_data_contract_questions` fills
    them."""
    return _DATA_CONTRACT_QUESTION_ROLE_PATH.read_text(encoding="utf-8")


def load_adr_generation_role() -> str:
    """`prompts/adr_generator.jinja`'s raw text, `{{transcript}}`/
    `{{clarifications}}` placeholders still unfilled — `build_adr_generation_prompt` fills
    them. Used by the production graph's `synthesize_document` node (`agents/graph.py`) and
    exercised in isolation by `testing_adr_acb/`'s golden set."""
    return _ADR_GENERATION_ROLE_PATH.read_text(encoding="utf-8")


def load_question_classifier_role() -> str:
    """`prompts/question_classifier.jinja`'s raw text, `{{questions}}`/`{{transcript}}`
    placeholders still unfilled — `build_classification_prompt` fills them. Used by the
    production graph's `classify_questions` node (`agents/graph.py`)."""
    return _QUESTION_CLASSIFIER_PATH.read_text(encoding="utf-8")


def load_adr_critic_role() -> str:
    """`prompts/adr_critic.jinja`'s raw text, `{{document}}`/`{{clarifications}}`/
    `{{source_content}}` placeholders still unfilled — `build_critic_prompt` fills them. Used
    by the production graph's `critic_document` node (`agents/graph.py`)."""
    return _ADR_CRITIC_PATH.read_text(encoding="utf-8")


def extract_json(content: str) -> str:
    """Strip a ```json fence if the model wrapped its structured output in one,
    despite being asked not to. A no-op on already-bare JSON."""
    return _FENCE_RE.sub("", content).strip()


def load_json_response(content: str) -> dict:
    return json.loads(extract_json(content))
