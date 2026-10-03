"""TECHNIQUE: Citation verification (grounding).

Problem: an LLM can cite a fact it was never shown, or cite the wrong version. A false
citation still sounds confident. Trusting it without a check breaks RAG faithfulness.

How: `answer_question`/`answer_evolution_question` ask the LLM for an answer plus a list of
citations (`ChatCitation`: `entity_type`, `entity_id`, `version`). `_verify_citations` checks
each citation against `rows`, the facts actually retrieved — no extra LLM call. A citation that
does not match a retrieved row is dropped before the user sees it. Same idea as
`agents.shared.mentions_grounded_in_source`, applied to chat citations instead of component
mentions.

Used by: `answer_question`, `answer_evolution_question`."""

from collections.abc import Sequence

from agents.stages.gold.schemas import ChatCitation
from db.models import GoldEvolution


def _verify_citations(citations: list[ChatCitation], rows: Sequence[GoldEvolution]) -> list[ChatCitation]:
    """Keeps only citations that match a row in `rows`. Drops any `(entity_type, entity_id,
    version)` the LLM claims but never retrieved. This turns a citation into evidence, not an
    unverified claim."""
    known = {(row.entity_type, row.entity_id, row.version) for row in rows}
    return [c for c in citations if (c.entity_type, c.entity_id, c.version) in known]
