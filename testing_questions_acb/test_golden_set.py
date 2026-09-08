"""Real-LLM golden-set collection: for each transcript in `golden_set/` (increasing
complexity from one clean component to a deliberately chaotic ten-plus-component mess),
runs the Actor — `agents.service.generate_questions_for_batch`, the exact function
`agents/graph.py`'s `generate_questions` node and `make questions` both call — and
records a summary of each run into `testing_questions_acb/output/result.json`. The full
drafted `mentioned_components`/`questions` for a case aren't duplicated there — they
already live in that case's own `output/ingestion_date=golden-<name>/questions/
<name>.json`, written by `generate_questions_for_batch` itself; `result.json` is the
at-a-glance table across the whole golden set, not a second copy of the same content.

Two independent things happen per case, and only one of them can fail the test:

  - **Free, deterministic checks** (`_check_structural_quality` /
    `_check_mentioned_components_are_grounded`) — mechanical integrity checks on the
    collection itself (empty output, duplicate ids, duplicate/malformed question text, a
    hallucinated component name), at zero LLM cost. `passed` in `result.json` is this
    outcome, and it's what the test actually asserts on.
  - **The Critic** — a second, independent LLM call scoring 0-100 how completely the
    drafted questions would document the architecture change if a human answered every
    one of them, plus a one-paragraph `reason`. Deliberately written against general
    principles (completeness, grounding), not against `clarification_questions.jinja`'s
    own internal structure — an earlier version of this rubric was tied to a specific,
    heavily-specified prompt and went stale the moment the prompt was rewritten. `score`/
    `reason` are recorded for information only; they do **not** gate the test. A
    probabilistic judge deciding pass/fail turned this suite flaky before — see git
    history — so this time only the deterministic checks above do that.

Deliberately excluded from the default `pytest`/`make test` run (`testpaths = ["tests"]`
in pyproject.toml) — this hits the real LLM twice per case (Actor + Critic), on purpose,
to see the actual prompt's real-world output, not a mocked one.

Run:
    uv run pytest testing_questions_acb/ -v
    MIN_QUESTIONS=15 uv run pytest testing_questions_acb/ -v   # stricter floor

Needs a real OPENAI_API_KEY/ANTHROPIC_API_KEY (same as any other real run against
llm.router.complete) — no server, no Postgres, no LangGraph needed.

Every case's `{name, question_count, passed, score, reason, input_tokens, output_tokens,
estimated_euro_cost}` is written to `testing_questions_acb/output/result.json` once the
whole run finishes (`conftest.py`'s `pytest_sessionfinish`), alongside the same
per-transcript question JSON files a real `generate_questions` run would produce. That
file's content is then printed to the terminal via `conftest.py`'s
`pytest_terminal_summary` hook, visible without needing `-s`.

Cost tracking: every real LLM call this test makes (Actor + Critic) is metered via a
`litellm.acompletion` wrapper (`_track_llm_cost` fixture) that sums input/output tokens
and USD cost (`litellm.completion_cost`, its own maintained per-model pricing table),
converted to a rough EUR estimate at a fixed `USD_TO_EUR` rate — not a live FX lookup,
just enough to see the order of magnitude of running this suite.
"""

import os
from pathlib import Path

import litellm
import pytest
from pydantic import BaseModel

from agents.schemas import MentionedComponent, QuestionItem
from agents.service import generate_questions_for_batch
from agents.template import load_json_response
from app.config import settings
from llm import router

from testing_questions_acb.conftest import record_result

MIN_QUESTIONS = int(os.environ.get("MIN_QUESTIONS", "2"))
CRITIC_PROVIDER_ORDER = list(reversed(settings.llm_fallback_order))
USD_TO_EUR = 0.92

GOLDEN_SET_DIR = Path(__file__).parent / "golden_set"

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def _track_llm_cost(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Wraps the real `litellm.acompletion` (not a mock — the call still hits the real API)
    to meter every LLM call this test case makes: Actor + Critic. Function-scoped, so each
    parametrized golden-set case gets its own fresh counters."""
    usage = {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
    original_acompletion = litellm.acompletion

    async def metered_acompletion(*args, **kwargs):
        response = await original_acompletion(*args, **kwargs)
        if getattr(response, "usage", None) is not None:
            usage["input_tokens"] += response.usage.prompt_tokens
            usage["output_tokens"] += response.usage.completion_tokens
        try:
            usage["cost_usd"] += litellm.completion_cost(completion_response=response)
        except Exception:  # noqa: BLE001 - pricing lookup can fail for an unlisted model;
            pass  # cost tracking must never fail the actual test over this.
        return response

    monkeypatch.setattr(litellm, "acompletion", metered_acompletion)
    return usage


class EvaluationResult(BaseModel):
    score: int
    rationale: str


def _check_structural_quality(questions: list[QuestionItem]) -> list[str]:
    """Deterministic, LLM-free integrity checks. Returns violation messages (empty =
    clean) — mechanical breakage only (empty output, duplicate ids, duplicate/malformed
    question text), never a judgment about whether the questions are good."""
    violations = []
    if len(questions) < MIN_QUESTIONS:
        violations.append(f"only {len(questions)} question(s) drafted (< {MIN_QUESTIONS})")

    ids = [q.id for q in questions]
    duplicate_ids = sorted({i for i in ids if ids.count(i) > 1})
    if duplicate_ids:
        violations.append(f"{len(duplicate_ids)} duplicate id(s): {duplicate_ids}")

    normalized = [q.question.strip().lower() for q in questions]
    duplicate_text = sorted({q for q in normalized if normalized.count(q) > 1})
    if duplicate_text:
        violations.append(f"{len(duplicate_text)} duplicate question(s): {duplicate_text}")

    malformed = [q.question for q in questions if "?" not in q.question]
    if malformed:
        violations.append(f"{len(malformed)} question(s) have no '?' at all: {malformed}")

    return violations


def _check_mentioned_components_are_grounded(
    mentioned_components: list[MentionedComponent], transcript: str
) -> list[str]:
    """Deterministic, LLM-free check: every claimed `mentioned_components` entry must
    appear in the transcript verbatim (case-insensitive) — catches an invented or
    paraphrased component name mechanically.
    """
    violations = []
    if not mentioned_components:
        violations.append("mentioned_components is empty — no component identified at all")

    transcript_lower = transcript.lower()
    ungrounded = [c.name for c in mentioned_components if c.name.lower() not in transcript_lower]
    if ungrounded:
        violations.append(f"{len(ungrounded)} mentioned_components not found in the transcript: {ungrounded}")

    return violations


def _golden_set_cases() -> list[tuple[str, str]]:
    """[(name, transcript_text), ...], sorted by filename so 01_.. runs before 10_.."""
    return [(f.stem, f.read_text(encoding="utf-8")) for f in sorted(GOLDEN_SET_DIR.glob("*.txt"))]


def _bronze_documents_for(name: str, transcript: str) -> list[dict]:
    return [{"source_component": f"{name}.en.vtt", "content": transcript}]


def _questions_block(questions: list[QuestionItem]) -> str:
    return "\n".join(f"- [{q.scope}] {q.question}" for q in questions) or "(no questions were drafted)"


def _build_evaluation_prompt(transcript: str, questions: list[QuestionItem]) -> list[dict]:
    content = (
        "You are grading a list of architecture-change clarification questions drafted for the "
        "meeting transcript below.\n\n"
        "Score from 0 to 100: if a human who was in the meeting answered every one of these "
        "questions, would the architecture change described in the transcript end up fully and "
        "accurately documented — every component, every interaction, every data contract, and "
        "every decision the transcript actually raises?\n\n"
        "0 means answering them would leave the change about as undocumented as the raw "
        "transcript already is. 100 means answering them would leave nothing material the "
        "transcript raises still ambiguous.\n\n"
        "Penalize two things specifically: something the transcript clearly raises but no "
        "question covers at all, and padding — a question about something the transcript never "
        "actually raised.\n\n"
        f"Transcript:\n{transcript}\n\n"
        f"Drafted questions:\n{_questions_block(questions)}\n\n"
        'Return JSON exactly like {"score": <integer 0-100>, "rationale": "<one paragraph '
        'explaining what the questions cover well and what they miss or pad>"}. Nothing else.'
    )
    return [{"role": "user", "content": content}]


async def _evaluate(transcript: str, questions: list[QuestionItem]) -> EvaluationResult:
    messages = _build_evaluation_prompt(transcript, questions)
    response = await router.complete(
        messages,
        providers=CRITIC_PROVIDER_ORDER,
        response_format=EvaluationResult,
        temperature=0,
    )
    return EvaluationResult.model_validate(load_json_response(response.choices[0].message.content))


@pytest.mark.parametrize(
    "name,transcript", _golden_set_cases(), ids=[name for name, _ in _golden_set_cases()]
)
async def test_golden_set_question_collection(name: str, transcript: str, _track_llm_cost: dict) -> None:
    # Actor — the exact function `generate_questions` (the graph node) and `make
    # questions` both call; also writes ingestion_date=<date>/questions/<name>.json
    # under this directory's own output/ (conftest.py redirects settings.output_dir).
    ingestion_date = f"golden-{name}"
    result = await generate_questions_for_batch(ingestion_date, _bronze_documents_for(name, transcript))

    violations = _check_structural_quality(result.questions)
    violations += _check_mentioned_components_are_grounded(result.mentioned_components, transcript)

    if violations:
        # Broken/near-empty output isn't worth spending a Critic call to score.
        score, reason = 0, "Structural check failed: " + "; ".join(violations)
    else:
        evaluation = await _evaluate(transcript, result.questions)
        score, reason = evaluation.score, evaluation.rationale

    record_result(
        name,
        len(result.questions),
        not violations,
        score,
        reason,
        _track_llm_cost["input_tokens"],
        _track_llm_cost["output_tokens"],
        round(_track_llm_cost["cost_usd"] * USD_TO_EUR, 6),
    )

    assert not violations, f"{name}: structural check failed: {violations}"
