"""Structured-output schemas for the architecture-questions stage. This is stage 1 of 2 in
the question-generation pipeline. It identifies components and data contracts, and it drafts
the clarification questions needed to resolve everything `prompts/architecture_questions/`
does not know yet. It never produces the full ODCS detail for a data contract. That is
`agents.stages.data_contract_questions`'s job, one stage later."""

from pydantic import BaseModel

from agents.shared import ComponentStatus, ContractAction, QuestionItem


class MentionedComponent(BaseModel):
    """One entry of `ArchitectureQuestionListResult.mentioned_components`. This is a
    component the transcript names, plus the Actor's own read of its lifecycle status
    relative to `KNOWN_ARCHITECTURE`. See prompts/architecture_questions/combined.jinja,
    COMPONENT STATUS."""

    name: str
    status: ComponentStatus


class MentionedDataContract(BaseModel):
    """One entry of `ArchitectureQuestionListResult.mentioned_data_contracts`. This is a data
    contract that this stage identified. It only has the name, producer, consumer, and action,
    never the full schema. This is the fixed list that
    `prompts/data_contract_questions/questions.jinja`'s `IDENTIFIED_DATA_CONTRACTS` input is
    built from. The data-contract stage never discovers a contract on its own. Only this stage
    does."""

    name: str
    producer: str
    consumer: str
    action: ContractAction


class ArchitectureQuestionListResult(BaseModel):
    """`generate_architecture_questions_for_batch`'s output. See
    prompts/architecture_questions/combined.jinja. This is question-generation stage 1 of 2:
    ADR, components, and data-contract identification. It never produces the full ODCS detail.
    That is `agents.stages.data_contract_questions.schemas.DataContractQuestionListResult`'s
    job. It has exactly three top-level fields. The prompt itself says clearly that no other
    fields are allowed.

    We check `mentioned_components` mechanically against the transcript. We do not just trust
    it. Each name must appear in the transcript word for word. This catches an invented or
    paraphrased component name in a reliable way, without needing an LLM judge.
    `mentioned_data_contracts` feeds directly into stage 2 as its fixed
    `IDENTIFIED_DATA_CONTRACTS` list.
    """

    mentioned_components: list[MentionedComponent]
    mentioned_data_contracts: list[MentionedDataContract]
    questions: list[QuestionItem]


# --- Decomposed architecture-questions pipeline. This is an evaluation prototype. It is not
# wired into agents/graph.py yet. See TESTING_REPORT.md's advice on splitting
# architecture_questions.jinja's single prompt of about 1450 lines into smaller calls that we
# can test one at a time. There are three schemas, one per call. Together they mirror
# ArchitectureQuestionListResult's own three fields, but three separate LLM calls produce them
# instead of one: identification (this file's own MentionedComponent and
# MentionedDataContract, no questions at all), drafting (a broad candidate `questions` list,
# not filtered yet), and selection (the same shape, filtered and deduplicated down to the
# final set). We keep these as their own named types instead of reusing
# DataContractQuestionListResult, even though it happens to share the same
# `{questions: [...]}` shape. This way each call site's intent is clear on its own. It does
# not rely on a name borrowed from an unrelated stage. -----------------------------


class ArchitectureIdentificationResult(BaseModel):
    """Call 1 of 3 (`prompts/architecture_questions/identification.jinja`). This call only
    identifies things. It never drafts questions. Every `"unknown"` field here, such as a
    component's status or a contract's action, producer, or consumer, is exactly what call 2
    turns into a clarification question. This call's only job is to produce a complete,
    grounded inventory for call 2 to work from."""

    mentioned_components: list[MentionedComponent]
    mentioned_data_contracts: list[MentionedDataContract]


class DraftedQuestionsResult(BaseModel):
    """Call 2 of 3 (`prompts/architecture_questions/drafting.jinja`). This call builds a
    deliberately broad, unfiltered candidate list from call 1's identification result plus the
    transcript. Selecting the best questions, such as removing duplicates, rejecting trivial
    ones, and scoring value, is call 3's job, not this one's. This call should "cast a wide
    net," per the original prompt's own SELECTION STRATEGY step 2."""

    questions: list[QuestionItem]


class SelectedQuestionsResult(BaseModel):
    """Call 3 of 3 (`prompts/architecture_questions/selection.jinja`). This call filters call
    2's candidates down to the small, high-value final set that an architect should actually
    be asked. It has the same shape as `DraftedQuestionsResult`, a plain `questions` list, but
    we keep it as its own type. A function that returns this type is saying "these are already
    selected." A function that returns the other type is saying "these still need to be
    selected.\""""

    questions: list[QuestionItem]
