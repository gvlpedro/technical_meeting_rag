"""Builds the prompt for the Gold stage's one LLM call. The rest of
`agents.stages.gold.service` — identity resolution, persistence — is deterministic."""

import jinja2

from agents.template import PROMPTS_DIR

_ROLE_PATH = PROMPTS_DIR / "gold" / "extraction.jinja"


def load_gold_extraction_role() -> str:
    """Returns the raw text of `prompts/gold/extraction.jinja`, placeholders unfilled.
    `build_gold_extraction_prompt` fills them in."""
    return _ROLE_PATH.read_text(encoding="utf-8")


def build_gold_extraction_prompt(adr_content: str) -> list[dict]:
    """Renders `prompts/gold/extraction.jinja`. One LLM call per source, run right after the ADR
    is approved. Takes only the final, clarified ADR — Gold discovers every component and
    contract from the ADR text, not from an earlier, pre-clarification list. An earlier design
    used that earlier list and missed a component introduced only through a clarification
    answer."""
    role_template = jinja2.Template(load_gold_extraction_role())
    prompt = role_template.render(adr_content=adr_content)
    return [{"role": "user", "content": prompt}]
