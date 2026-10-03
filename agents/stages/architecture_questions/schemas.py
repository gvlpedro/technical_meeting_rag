"""Structured-output schemas for the architecture-questions stage (question-generation stage
1 of 2). Identifies components and data contracts, and drafts clarification questions. Full
ODCS detail is `agents.stages.data_contract_questions`'s job, one stage later."""

from pydantic import BaseModel

from agents.shared import ComponentStatus, ContractAction, QuestionItem


class MentionedComponent(BaseModel):
    """A component the transcript names, plus its lifecycle status relative to
    `KNOWN_ARCHITECTURE`. See combined.jinja, COMPONENT STATUS."""

    name: str
    status: ComponentStatus


class MentionedDataContract(BaseModel):
    """A data contract this stage identified: name, producer, consumer, action, never the full
    schema. Feeds `IDENTIFIED_DATA_CONTRACTS` in the data-contract stage, which never discovers
    a contract on its own."""

    name: str
    producer: str
    consumer: str
    action: ContractAction


class ArchitectureQuestionListResult(BaseModel):
    """`generate_architecture_questions_for_batch`'s output: components, data contracts,
    and clarification questions. Exactly three fields; no others allowed.

    `mentioned_components` is checked mechanically against the transcript — each name must
    appear word for word, catching an invented or paraphrased name without an LLM judge.
    `mentioned_data_contracts` feeds stage 2 as its fixed `IDENTIFIED_DATA_CONTRACTS` list."""

    mentioned_components: list[MentionedComponent]
    mentioned_data_contracts: list[MentionedDataContract]
    questions: list[QuestionItem]


# --- Decomposed architecture-questions pipeline (evaluation prototype, not wired into
# agents/graph.py). Splits one big prompt into three calls: identification, drafting
# (broad, unfiltered candidates), and selection (filtered, deduplicated final set). Separate
# named types, even where the shape matches an existing one, so each call's intent is clear. --


class ArchitectureIdentificationResult(BaseModel):
    """Call 1 of 3 (identification.jinja). Identifies things; never drafts questions. Each
    `"unknown"` field here is what call 2 turns into a clarification question."""

    mentioned_components: list[MentionedComponent]
    mentioned_data_contracts: list[MentionedDataContract]


class DraftedQuestionsResult(BaseModel):
    """Call 2 of 3 (drafting.jinja). Builds a broad, unfiltered candidate list from call 1's
    result. Selecting the best questions is call 3's job, not this one's."""

    questions: list[QuestionItem]


class SelectedQuestionsResult(BaseModel):
    """Call 3 of 3 (selection.jinja). Filters call 2's candidates to the final, high-value set.
    Same shape as `DraftedQuestionsResult`, but a separate type: this one means "selected"."""

    questions: list[QuestionItem]
