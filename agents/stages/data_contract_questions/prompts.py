"""Prompt construction for the data-contract-questions stage. It has one `.jinja` file, one
path constant, one loader, and one builder function."""

import jinja2

from agents.state import MentionedDataContractItem
from agents.template import PROMPTS_DIR

_STAGE_DIR = PROMPTS_DIR / "data_contract_questions"
_ROLE_PATH = _STAGE_DIR / "questions.jinja"
_TEMPLATE_PATH = _STAGE_DIR / "template.md"


def load_data_contract_template() -> str:
    """`prompts/data_contract_questions/template.md`'s raw text. This is the per-contract ODCS
    completeness spec that this stage drafts its clarification questions against. It is
    applied once per contract in `IDENTIFIED_DATA_CONTRACTS`."""
    return _TEMPLATE_PATH.read_text(encoding="utf-8")


def load_data_contract_question_generation_role() -> str:
    """`prompts/data_contract_questions/questions.jinja`'s raw text. Its
    `{{data_contract_requirements}}`, `{{transcript}}`, and `{{identified_data_contracts}}`
    placeholders are still unfilled. `agents.graph`'s `generate_data_contract_questions` node
    fills them in via `build_data_contract_question_generation_prompt`."""
    return _ROLE_PATH.read_text(encoding="utf-8")


def _mentioned_data_contracts_block(contracts: list[MentionedDataContractItem]) -> str:
    if not contracts:
        return "(none identified yet)"
    return "\n".join(f"- {c['name']}: {c['producer']} -> {c['consumer']} ({c['action']})" for c in contracts)


def build_data_contract_question_generation_prompt(
    template_text: str, transcript_text: str, mentioned_data_contracts: list[MentionedDataContractItem]
) -> list[dict]:
    """Renders `prompts/data_contract_questions/questions.jinja`. This is question-generation
    stage 2 of 2: full ODCS-completeness questions for every contract that stage 1
    (`agents.stages.architecture_questions.prompts.build_architecture_question_generation_
    prompt`) already identified. This stage never discovers a contract on its own. It only
    drafts questions for the fixed list it is given."""
    role_template = jinja2.Template(load_data_contract_question_generation_role())
    prompt = role_template.render(
        data_contract_requirements=template_text,
        transcript=transcript_text,
        identified_data_contracts=_mentioned_data_contracts_block(mentioned_data_contracts),
    )
    return [{"role": "user", "content": prompt}]
