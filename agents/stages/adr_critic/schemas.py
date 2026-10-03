"""Structured-output schema for the ADR-critic stage. It reviews a drafted ADR against its
transcript and clarifications, flags unsupported claims, and scores completeness. This stage
always runs (`agents.graph.critic_document`)."""

from typing import Literal

from pydantic import BaseModel


class CritiqueClaim(BaseModel):
    # An unsupported claim downgrades the confidence level.
    claim: str
    supported: bool
    rationale: str
    severity: Literal["low", "material"]


class CritiqueResult(BaseModel):
    claims: list[CritiqueClaim]
    # Score 0-100. See prompts/adr_critic/critic.jinja, COMPLETENESS SCORING.
    completeness_score: int
    unresolved_points: list[str] = []
