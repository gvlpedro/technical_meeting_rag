"""This file builds the prompt for the classification stage. The stage has one `.jinja` file,
one path constant, one loader function, and one builder function.

The real classify call lives in `agents.graph.classify_questions`. That call uses
`temperature=0` and matches results back to `agents.state.ClarificationItem` by `id`.

This stage package has no separate `service.py`. Classification has no LLM call worth pulling
out of the graph node. This is different from the two question-generation stages. Both of those
stages have real standalone callers: `make questions`, their own golden sets, and the frontend's
"ask me more" endpoint."""

import jinja2

from agents.state import GeneratedQuestion
from agents.template import PROMPTS_DIR

_ROLE_PATH = PROMPTS_DIR / "classification" / "classifier.jinja"


def load_question_classifier_role() -> str:
    """Returns the raw text of `prompts/classification/classifier.jinja`. The `{{questions}}`
    and `{{transcript}}` placeholders are still empty here. `build_classification_prompt` fills
    them in. The production graph's `classify_questions` node uses this (`agents/graph.py`)."""
    return _ROLE_PATH.read_text(encoding="utf-8")


def _generated_questions_block(questions: list[GeneratedQuestion]) -> str:
    return "\n".join(f"- id: {q['id']}\n  question: {q['question']}" for q in questions)


def build_classification_prompt(questions: list[GeneratedQuestion], transcript_text: str) -> list[dict]:
    """Renders `prompts/classification/classifier.jinja` as a real Jinja template. This follows
    the same single-user-message convention every prompt builder in this codebase uses. It does
    not build a separate system and user pair. That split used to live only in this function's
    own Python string. Now the instructions live in the template's own opening section instead."""
    role_template = jinja2.Template(load_question_classifier_role())
    prompt = role_template.render(
        questions=_generated_questions_block(questions),
        transcript=transcript_text,
    )
    return [{"role": "user", "content": prompt}]
