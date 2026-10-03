"""Prompt construction for the data-contract-questions stage."""

import jinja2

from agents.state import MentionedDataContractItem
from agents.template import PROMPTS_DIR

_STAGE_DIR = PROMPTS_DIR / "data_contract_questions"
_ROLE_PATH = _STAGE_DIR / "questions.jinja"
_TEMPLATE_PATH = _STAGE_DIR / "template.md"


def load_data_contract_template() -> str:
    """`template.md`'s raw text: the per-contract ODCS completeness spec, applied once per
    contract in `IDENTIFIED_DATA_CONTRACTS`."""
    return _TEMPLATE_PATH.read_text(encoding="utf-8")


def load_data_contract_question_generation_role() -> str:
    """`questions.jinja`'s raw text, placeholders unfilled.
    `build_data_contract_question_generation_prompt` fills them in."""
    return _ROLE_PATH.read_text(encoding="utf-8")


def _mentioned_data_contracts_block(contracts: list[MentionedDataContractItem]) -> str:
    if not contracts:
        return "(none identified yet)"
    return "\n".join(f"- {c['name']}: {c['producer']} -> {c['consumer']} ({c['action']})" for c in contracts)


def build_data_contract_question_generation_prompt(
    template_text: str, transcript_text: str, mentioned_data_contracts: list[MentionedDataContractItem]
) -> list[dict]:
    """Renders `questions.jinja`: ODCS-completeness questions for contracts stage 1 already
    identified. This stage never discovers a contract on its own."""
    role_template = jinja2.Template(load_data_contract_question_generation_role())
    prompt = role_template.render(
        data_contract_requirements=template_text,
        transcript=transcript_text,
        identified_data_contracts=_mentioned_data_contracts_block(mentioned_data_contracts),
    )
    return [{"role": "user", "content": prompt}]
