"""Structured-output schema for the data-contract-questions stage (question-generation
stage 2 of 2). Drafts full ODCS v3.0.0-completeness questions for the contracts
`agents.stages.architecture_questions` already identified."""

from pydantic import BaseModel

from agents.shared import QuestionItem


class DataContractQuestionListResult(BaseModel):
    """`generate_data_contract_questions_for_batch`'s output. Every `questions[].scope` is
    `data_contract`, and every `target` must match one of the given contracts — checked
    mechanically, same as stage 1's component grounding check."""

    questions: list[QuestionItem]
