"""Structured-output schemas for the LLM calls that need one.

`synthesize_document` deliberately has no schema here — it writes the clarified
document directly as Markdown (`doc/silver_process.md` §3 node 5: "no intermediate
structured JSON"), so its response is used as raw text, not parsed against a model.
"""

from typing import Literal

from pydantic import BaseModel

ComponentStatus = Literal["new", "modified", "removed", "unchanged", "unknown"]
QuestionScope = Literal[
    "metadata", "component", "architecture", "data_contract", "adr", "change_impact", "migration"
]


class MentionedComponent(BaseModel):
    """One entry of `QuestionListResult.mentioned_components` — a component the transcript
    names, plus the Actor's own read of its lifecycle status relative to `KNOWN_ARCHITECTURE`
    (see prompting/roles/common/clarification_questions.jinja, COMPONENT STATUS)."""

    name: str
    status: ComponentStatus


class QuestionItem(BaseModel):
    """One drafted clarification question — carries a stable `id` (see the prompt's
    QUESTION IDENTIFIERS section) so downstream steps (`classify_questions`) can match a
    classification back to its question without relying on exact-text equality."""

    id: str
    scope: QuestionScope
    target: str
    requirement: str
    question: str


class QuestionListResult(BaseModel):
    """`generate_questions`'s output — see prompting/roles/common/clarification_questions.jinja.
    Exactly two top-level fields — the prompt itself is explicit that no others are allowed.

    `mentioned_components` is checked mechanically (not just trusted) against the transcript by
    `testing_questions_acb`'s golden-set test — each name must appear in the transcript verbatim,
    catching an invented or paraphrased component name deterministically, without an LLM judge.
    """

    mentioned_components: list[MentionedComponent]
    questions: list[QuestionItem]


class QuestionClassification(BaseModel):
    # Matched back to its question by `id`, not by re-sending/comparing question text — the
    # classifier only ever needs to echo the id it was given.
    id: str
    answer: str | None = None
    status: Literal["answered", "unknown", "needs_clarification"]


class ClassificationResult(BaseModel):
    classifications: list[QuestionClassification]


class CritiqueClaim(BaseModel):
    # Verbatim substring copied from the drafted document, not paraphrased — this is
    # what lets boss_decide locate and downgrade it in-place later.
    claim: str
    supported: bool
    rationale: str
    severity: Literal["low", "material"]


class CritiqueResult(BaseModel):
    claims: list[CritiqueClaim]
