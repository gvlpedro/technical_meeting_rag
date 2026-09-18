"""This file builds the prompt for the Gold stage. The stage has one `.jinja` file, one path
constant, one loader function, and one builder function. This is the only LLM call in the whole
Gold stage. Identity resolution and persistence, the rest of `agents.stages.gold.service`, are
deterministic and use no LLM."""

import jinja2

from agents.template import PROMPTS_DIR

_ROLE_PATH = PROMPTS_DIR / "gold" / "extraction.jinja"


def load_gold_extraction_role() -> str:
    """Returns the raw text of `prompts/gold/extraction.jinja`. The `{{adr_content}}`,
    `{{mentioned_components}}`, and `{{mentioned_data_contracts}}` placeholders are still empty
    here. `build_gold_extraction_prompt` fills them in.

    `agents.stages.gold.service.extract_gold_facts_for_source` uses this. That function is
    called from the production graph's `extract_gold_facts` node (`agents/graph.py`). This runs
    inside Silver's own graph, right after `chunk_and_embed`, per `.tmp/gold_process_v5.md` §1-2.
    It does not run as a separately triggered pass."""
    return _ROLE_PATH.read_text(encoding="utf-8")


def build_gold_extraction_prompt(adr_content: str) -> list[dict]:
    """Renders `prompts/gold/extraction.jinja`. This is one LLM call per source, run inside
    Silver's own graph right after the ADR is approved (`.tmp/gold_process_v5.md` §1-2). It is
    not a separately triggered pass over `silver_documents` rows.

    This function takes only the final, clarified ADR. It does not take a pre-given
    `mentioned_components` or `mentioned_data_contracts` list. Gold discovers every component
    and contract directly from the ADR itself. The ADR is already the validated, enriched output
    of the clarification process.

    An earlier design grounded extraction against an EARLIER, pre-clarification list instead:
    the transcript-time identification the architecture-questions stage drafts. That approach had
    a problem. A component introduced only through a clarification answer, and never named in
    that earlier list, could never reach Gold at all, even though the final ADR plainly described
    it."""
    role_template = jinja2.Template(load_gold_extraction_role())
    prompt = role_template.render(adr_content=adr_content)
    return [{"role": "user", "content": prompt}]
