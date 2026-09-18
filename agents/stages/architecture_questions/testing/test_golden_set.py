"""This is a real-LLM golden-set suite. For each transcript in `golden_set/`, it runs the
Actor and checks the result. The transcripts grow in complexity, from one clean component
to a deliberately chaotic case with ten or more components.

The Actor is `agents.stages.architecture_questions.service.generate_architecture_questions_for_batch`. This is the exact
function that `agents/graph.py`'s `generate_architecture_questions` node and `make
questions` both call. This suite records a summary of each run into
`agents/stages/architecture_questions/testing/output/result.json`. The full drafted
`mentioned_components`/`questions` for a case are not duplicated there. They already live
in that case's own `output/ingestion_date=golden-<name>/questions/<name>.json` file,
written by `generate_architecture_questions_for_batch` itself. `result.json` is a table
you can scan at a glance across the whole golden set. It is not a second copy of the same
content.

Two independent things happen for each case. Only one of them can make the test fail:

  - **Free, deterministic checks** (`_check_structural_quality`,
    `_check_mentioned_components_are_grounded`, and
    `_check_mentioned_data_contracts_are_grounded`). These are mechanical integrity checks
    on the collection itself: empty output, duplicate ids, duplicate or malformed question
    text, or a hallucinated component or contract name. These checks cost nothing to run.
    `passed` in `result.json` reports this outcome, and this is what the test actually
    asserts on.
  - **The Critic.** This is a second, independent LLM call. It scores from 0 to 100 how
    completely the drafted questions would document the architecture change, if a human
    answered every one of them. It also writes a one-paragraph `reason`. Its prompt lives
    in this directory's own `critic_prompt.jinja`, rendered through
    `_build_evaluation_prompt`. This prompt is deliberately written against general
    principles, such as completeness and grounding. It is not written against
    `architecture_questions.jinja`'s own internal structure. An earlier version of this
    rubric was tied to one specific, heavily-specified prompt. That version went stale the
    moment the prompt was rewritten. `score` and `reason` are recorded for information
    only. They do **not** decide pass or fail. A probabilistic judge deciding pass or fail
    made this suite flaky before (see git history). So now only the deterministic checks
    above decide that.

This suite is deliberately excluded from the default `pytest`/`make test` run (see
`testpaths = ["tests"]` in pyproject.toml). It calls the real LLM twice per case, once for
the Actor and once for the Critic, on purpose. This way we see the prompt's real output,
not a mocked one.

Run:
    uv run pytest agents/stages/architecture_questions/testing/ -v
    MIN_QUESTIONS=15 uv run pytest agents/stages/architecture_questions/testing/ -v   # stricter floor

This suite needs a real OPENAI_API_KEY/ANTHROPIC_API_KEY, the same as any other real run
against llm.router.complete. It needs no server, no Postgres, and no LangGraph.

Each case's `{name, question_count, passed, score, reason, input_tokens, output_tokens,
estimated_euro_cost}` is written to `agents/stages/architecture_questions/testing/output/result.json` once
the whole run finishes, through `conftest.py`'s `pytest_sessionfinish` hook. This happens
alongside the same per-transcript question JSON files a real `generate_architecture_questions`
run would produce. That file's content is then printed to the terminal through
`conftest.py`'s `pytest_terminal_summary` hook, so you can see it without needing `-s`.

Cost tracking: every real LLM call this test makes, for both the Actor and the Critic, is
metered through a `litellm.acompletion` wrapper (the `_track_llm_cost` fixture). It sums
input and output tokens, and the USD cost from `litellm.completion_cost`, which keeps its
own per-model pricing table. This is converted to a rough EUR estimate at a fixed
`USD_TO_EUR` rate. This is not a live currency lookup. It is only enough to see the order
of magnitude of running this suite.
"""

import os
from pathlib import Path

import jinja2
import litellm
import pytest
from pydantic import BaseModel

from agents.shared import QuestionItem, name_appears_in_text
from agents.stages.architecture_questions.schemas import MentionedComponent, MentionedDataContract
from agents.stages.architecture_questions.service import generate_architecture_questions_for_batch
from agents.template import load_json_response
from app.config import settings
from llm import router

from agents.stages.architecture_questions.testing.conftest import record_result

MIN_QUESTIONS = int(os.environ.get("MIN_QUESTIONS", "2"))
CRITIC_PROVIDER_ORDER = list(reversed(settings.llm_fallback_order))
USD_TO_EUR = 0.92

GOLDEN_SET_DIR = Path(__file__).parent / "golden_set"
_CRITIC_PROMPT_PATH = Path(__file__).parent / "critic_prompt.jinja"

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def _track_llm_cost(monkeypatch: pytest.MonkeyPatch) -> dict:
    """This fixture wraps the real `litellm.acompletion` call. It is not a mock. The call
    still reaches the real API. It meters every LLM call this test case makes, for both the
    Actor and the Critic. It is function-scoped, so each parametrized golden-set case gets
    its own fresh counters."""
    usage = {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
    original_acompletion = litellm.acompletion

    async def metered_acompletion(*args, **kwargs):
        response = await original_acompletion(*args, **kwargs)
        if getattr(response, "usage", None) is not None:
            usage["input_tokens"] += response.usage.prompt_tokens
            usage["output_tokens"] += response.usage.completion_tokens
        try:
            usage["cost_usd"] += litellm.completion_cost(completion_response=response)
        except Exception:  # noqa: BLE001 - pricing lookup can fail for an unlisted model.
            pass  # Cost tracking must never make the actual test fail over this.
        return response

    monkeypatch.setattr(litellm, "acompletion", metered_acompletion)
    return usage


class EvaluationResult(BaseModel):
    score: int
    rationale: str


def _check_structural_quality(questions: list[QuestionItem]) -> list[str]:
    """These are deterministic integrity checks. They do not use the LLM. This function
    returns a list of violation messages. An empty list means the output is clean. These
    checks only catch mechanical breakage: empty output, duplicate ids, or duplicate or
    malformed question text. They never judge whether the questions are good."""
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
    """This check is deterministic and does not use the LLM. Every claimed
    `mentioned_components` entry must appear in the transcript word for word. The match is
    case-insensitive and word-boundary-safe (see `agents.shared.name_appears_in_text`).
    This catches an invented or paraphrased component name mechanically. Production's own
    `_ungrounded_component_names` uses this same building block. So a fix to the grounding
    rule lands in one place, not here and in production separately.
    """
    violations = []
    if not mentioned_components:
        violations.append("mentioned_components is empty — no component identified at all")

    transcript_lower = transcript.lower()
    ungrounded = [c.name for c in mentioned_components if not name_appears_in_text(c.name, transcript_lower)]
    if ungrounded:
        violations.append(f"{len(ungrounded)} mentioned_components not found in the transcript: {ungrounded}")

    return violations


def _check_mentioned_data_contracts_are_grounded(
    mentioned_data_contracts: list[MentionedDataContract], transcript: str
) -> list[str]:
    """This uses the same principle as `_check_mentioned_components_are_grounded`, applied
    to this stage's other identification output. Every claimed contract `name` must appear
    in the transcript word for word, case-insensitive and word-boundary-safe. This stage
    identifies contracts. It does not invent them. An invented one here would feed straight
    into the data-contract stage as if it were real.

    `"unknown"` is exempt. It is never a violation. `architecture_questions.jinja`'s PHASE
    6B explicitly requires that exact value for a contract implied only by a connection the
    transcript never named. That is the documented alternative to inventing a
    plausible-sounding name. It is not a hallucination this check exists to catch."""
    transcript_lower = transcript.lower()
    ungrounded = [
        c.name
        for c in mentioned_data_contracts
        if c.name != "unknown" and not name_appears_in_text(c.name, transcript_lower)
    ]
    if ungrounded:
        return [f"{len(ungrounded)} mentioned_data_contracts not found in the transcript: {ungrounded}"]
    return []


def _golden_set_cases() -> list[tuple[str, str]]:
    """Return [(name, transcript_text), ...], sorted by filename. This way, 01_.. runs
    before 10_.."""
    return [(f.stem, f.read_text(encoding="utf-8")) for f in sorted(GOLDEN_SET_DIR.glob("*.txt"))]


def _bronze_documents_for(name: str, transcript: str) -> list[dict]:
    return [{"source_component": f"{name}.en.vtt", "content": transcript}]


def _questions_block(questions: list[QuestionItem]) -> str:
    return "\n".join(f"- [{q.scope}] {q.question}" for q in questions) or "(no questions were drafted)"


def _load_critic_prompt_template() -> str:
    """Read the raw text of `agents/stages/architecture_questions/testing/critic_prompt.jinja`. This is the
    Critic's own prompt. It is kept as a real Jinja template here, not inline in this
    module. This way, you can edit and review it like any other prompt in this repo (see
    `prompts/*.jinja`). It is just scoped to this test suite instead of production."""
    return _CRITIC_PROMPT_PATH.read_text(encoding="utf-8")


def _build_evaluation_prompt(transcript: str, questions: list[QuestionItem]) -> list[dict]:
    template = jinja2.Template(_load_critic_prompt_template())
    content = template.render(transcript=transcript, questions=_questions_block(questions))
    return [{"role": "user", "content": content}]


async def _evaluate(transcript: str, questions: list[QuestionItem]) -> EvaluationResult:
    messages = _build_evaluation_prompt(transcript, questions)
    response = await router.complete(
        messages,
        providers=CRITIC_PROVIDER_ORDER,
        response_format=EvaluationResult,
        temperature=0,
        # This setting is required for a reasoning-locked OpenAI model (e.g. gpt-5.6-sol)
        # to accept temperature=0 at all. See the identical comment in agents/service.py.
        # Anthropic's own reasoning-locked models (e.g. claude-opus-5) have no equivalent
        # setting. So CRITIC_PROVIDER_ORDER's Anthropic-first attempt will now fail every
        # time, and fall through to OpenAI. This means the Critic silently ends up on the
        # same provider as the Actor it is judging. That defeats the reason this ordering
        # existed: to keep the two independent.
        reasoning_effort="none",
    )
    return EvaluationResult.model_validate(load_json_response(response.choices[0].message.content))


@pytest.mark.parametrize(
    "name,transcript", _golden_set_cases(), ids=[name for name, _ in _golden_set_cases()]
)
async def test_golden_set_question_collection(name: str, transcript: str, _track_llm_cost: dict) -> None:
    # This is the Actor step. It is the exact function `generate_architecture_questions`
    # (the graph node) and `make questions` both call. It also writes
    # ingestion_date=<date>/questions/<name>.json under this directory's own output/
    # folder. (conftest.py redirects settings.output_dir there.)
    ingestion_date = f"golden-{name}"
    result = await generate_architecture_questions_for_batch(ingestion_date, _bronze_documents_for(name, transcript))

    violations = _check_structural_quality(result.questions)
    violations += _check_mentioned_components_are_grounded(result.mentioned_components, transcript)
    violations += _check_mentioned_data_contracts_are_grounded(result.mentioned_data_contracts, transcript)

    if violations:
        # Broken or near-empty output is not worth spending a Critic call to score.
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
