"""This is the structured-output schema for the ADR-critic stage. This stage reviews a drafted
ADR against its source transcript and clarifications. It flags unsupported claims and scores
completeness. This stage is mandatory. It always runs. There is no confidence threshold that
skips it (`agents.graph.critic_document`)."""

from typing import Literal

from pydantic import BaseModel


class CritiqueClaim(BaseModel):
    # CLAIM: Not supported by the transcript, it will downgrade the confidence level
    claim: str
    supported: bool
    rationale: str
    severity: Literal["low", "material"]


class CritiqueResult(BaseModel):
    claims: list[CritiqueClaim]
    # A score from 0 to 100. See prompts/adr_critic/critic.jinja's COMPLETENESS SCORING
    # section. This reuses the Critic's own document read instead of making a second, dedicated
    # LLM call.
    completeness_score: int
    unresolved_points: list[str] = []
