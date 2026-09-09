"""Structured-output schemas for the LLM calls that need one.

`synthesize_document` deliberately has no schema here — it writes the clarified
document directly as Markdown (`doc/silver_process.md` §3 node 5: "no intermediate
structured JSON"), so its response is used as raw text, not parsed against a model.
"""

from typing import Literal

from pydantic import BaseModel

ComponentStatus = Literal["new", "modified", "removed", "unchanged", "unknown"]
ContractAction = Literal["new", "modified", "unchanged", "deprecated", "removed", "unknown"]
QuestionScope = Literal[
    "metadata", "component", "architecture", "data_contract", "adr", "change_impact", "migration"
]


class MentionedComponent(BaseModel):
    """One entry of `ArchitectureQuestionListResult.mentioned_components` — a component the
    transcript names, plus the Actor's own read of its lifecycle status relative to
    `KNOWN_ARCHITECTURE` (see prompts/architecture_questions.jinja, COMPONENT
    STATUS)."""

    name: str
    status: ComponentStatus


class MentionedDataContract(BaseModel):
    """One entry of `ArchitectureQuestionListResult.mentioned_data_contracts` — a data contract
    the architecture stage identified (name/producer/consumer/action only, never full schema).
    This is the fixed list `prompts/data_contract_questions.jinja`'s
    `IDENTIFIED_DATA_CONTRACTS` input is built from — the data-contract stage never discovers a
    contract on its own, only this stage does."""

    name: str
    producer: str
    consumer: str
    action: ContractAction


class QuestionItem(BaseModel):
    """One drafted clarification question — carries a stable `id` (see the prompt's
    QUESTION IDENTIFIERS section) so downstream steps (`classify_questions`) can match a
    classification back to its question without relying on exact-text equality."""

    id: str
    scope: QuestionScope
    target: str
    requirement: str
    question: str


class ArchitectureQuestionListResult(BaseModel):
    """`generate_architecture_questions`'s output — see
    prompts/architecture_questions.jinja (question-generation stage 1 of 2: ADR,
    components, and data-contract *identification* — never full ODCS detail, that's
    `DataContractQuestionListResult`'s job). Exactly three top-level fields — the prompt itself
    is explicit that no others are allowed.

    `mentioned_components` is checked mechanically (not just trusted) against the transcript —
    each name must appear in the transcript verbatim, catching an invented or paraphrased
    component name deterministically, without an LLM judge. `mentioned_data_contracts` feeds
    directly into stage 2 as its fixed `IDENTIFIED_DATA_CONTRACTS` list.
    """

    mentioned_components: list[MentionedComponent]
    mentioned_data_contracts: list[MentionedDataContract]
    questions: list[QuestionItem]


class DataContractQuestionListResult(BaseModel):
    """`generate_data_contract_questions`'s output — see
    prompts/data_contract_questions.jinja (question-generation stage 2 of 2: full
    ODCS-completeness questions for each contract `ArchitectureQuestionListResult.
    mentioned_data_contracts` already identified). Every `questions[].scope` is `data_contract`
    and every `questions[].target` must match one of the contracts it was given — checked
    mechanically, same principle as stage 1's component grounding check."""

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
