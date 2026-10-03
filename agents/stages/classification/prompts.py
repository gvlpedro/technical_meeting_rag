"""Builds the prompt for the classification stage. The real call is
`agents.graph.classify_questions`, which matches results back to `ClarificationItem` by `id`.
No separate `service.py` — unlike the question-generation stages, this one has no standalone
caller."""

import jinja2

from agents.state import GeneratedQuestion
from agents.template import PROMPTS_DIR

_ROLE_PATH = PROMPTS_DIR / "classification" / "classifier.jinja"


def load_question_classifier_role() -> str:
    """Returns the raw text of `prompts/classification/classifier.jinja`, placeholders unfilled.
    `build_classification_prompt` fills them in."""
    return _ROLE_PATH.read_text(encoding="utf-8")


def _generated_questions_block(questions: list[GeneratedQuestion]) -> str:
    return "\n".join(f"- id: {q['id']}\n  question: {q['question']}" for q in questions)


def build_classification_prompt(questions: list[GeneratedQuestion], transcript_text: str) -> list[dict]:
    """Renders `prompts/classification/classifier.jinja` as a single user message."""
    role_template = jinja2.Template(load_question_classifier_role())
    prompt = role_template.render(
        questions=_generated_questions_block(questions),
        transcript=transcript_text,
    )
    return [{"role": "user", "content": prompt}]
