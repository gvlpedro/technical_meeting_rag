"""Structured-output schema for the classification stage. Status `answered` drops a question.
Status `needs_clarification` or `unknown` keeps it. See `prompts/classification/classifier.jinja`
for the criteria."""

from typing import Literal

from pydantic import BaseModel


class QuestionClassification(BaseModel):
    # Links back to the question by id. The classifier only echoes the id given.
    id: str
    answer: str | None = None
    status: Literal["answered", "unknown", "needs_clarification"]
    # Id of the other question this duplicates, if any (see classifier.jinja, DUPLICATE
    # QUESTIONS). classify_questions folds duplicates together so a human sees each once.
    duplicate_of: str | None = None


class ClassificationResult(BaseModel):
    classifications: list[QuestionClassification]
