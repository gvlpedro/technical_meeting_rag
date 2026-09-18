"""Structured-output schema for the data-contract-questions stage. This is question-
generation stage 2 of 2 in the pipeline. It drafts full ODCS v3.0.0-completeness questions for
exactly the contracts that `agents.stages.architecture_questions` already identified."""

from pydantic import BaseModel

from agents.shared import QuestionItem


class DataContractQuestionListResult(BaseModel):
    """`generate_data_contract_questions_for_batch`'s output. See
    prompts/data_contract_questions/questions.jinja. This is question-generation stage 2 of 2:
    full ODCS-completeness questions for each contract that
    `agents.stages.architecture_questions.schemas.ArchitectureQuestionListResult.
    mentioned_data_contracts` already identified. Every `questions[].scope` is `data_contract`,
    and every `questions[].target` must match one of the contracts it was given. We check this
    mechanically, the same principle as stage 1's component grounding check."""

    questions: list[QuestionItem]
