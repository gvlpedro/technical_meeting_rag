"""This is the structured-output schema for the ADR-critic stage. This stage reviews a drafted
ADR against its source transcript and clarifications. It flags unsupported claims and scores
completeness. This stage is mandatory. It always runs. There is no confidence threshold that
skips it (`agents.graph.critic_document`)."""

from typing import Literal

from pydantic import BaseModel


class CritiqueClaim(BaseModel):
    # This is a verbatim substring copied from the drafted document. It is not paraphrased.
    # This is what lets `agents.graph.boss_decide` find it and downgrade it in place later.
    claim: str
    supported: bool
    rationale: str
    severity: Literal["low", "material"]


class CritiqueResult(BaseModel):
    claims: list[CritiqueClaim]
    # A score from 0 to 100. See prompts/adr_critic/critic.jinja's COMPLETENESS SCORING
    # section. This reuses the Critic's own document read instead of making a second, dedicated
    # LLM call. The Critic already reads the full document, transcript, and clarifications to
    # judge claim support. Completeness uses that same read, just applied to a different
    # question: "is anything left as a placeholder?"
    completeness_score: int
    unresolved_points: list[str] = []
