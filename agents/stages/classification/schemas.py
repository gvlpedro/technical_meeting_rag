"""This is the structured-output schema for the classification stage. It holds the one decision
that controls every clarification question this graph run can show to a human. A status of
`answered` drops a question silently. A status of `needs_clarification` or `unknown` keeps the
question. See `prompts/classification/classifier.jinja` for the real classification criteria."""

from typing import Literal

from pydantic import BaseModel


class QuestionClassification(BaseModel):
    # This field links back to its question by `id`. It does not resend or compare the
    # question text. The classifier only needs to echo back the id it was given.
    id: str
    answer: str | None = None
    status: Literal["answered", "unknown", "needs_clarification"]
    # Set only when this question asks for the same information as another question in the
    # same batch, just worded differently — see `prompts/classification/classifier.jinja`'s
    # DUPLICATE QUESTIONS section. Holds that other question's `id`. `None` means this
    # question stands on its own. `agents.graph.classify_questions` uses this to fold the
    # duplicate into the other question, so a human is never asked the same thing twice.
    duplicate_of: str | None = None


class ClassificationResult(BaseModel):
    classifications: list[QuestionClassification]
