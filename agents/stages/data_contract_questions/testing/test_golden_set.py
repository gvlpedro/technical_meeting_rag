"""This is a real-LLM golden-set suite. Each case in `golden_set/` has a transcript plus a
hand-authored `contracts.json`. This file holds the `mentioned_data_contracts` a real
architecture-stage run would have identified for that transcript. For each case, this
suite runs the Actor and records a summary of the run.

The Actor is `agents.stages.data_contract_questions.service.generate_data_contract_questions_for_batch`. This is the
exact function that `agents/graph.py`'s `generate_data_contract_questions` node and `make
questions` both call. This suite records a summary of each run into
`agents/stages/data_contract_questions/testing/output/result.json`.

This suite is deliberately its own project. It is not folded back into
`agents/stages/architecture_questions/testing/`'s golden set. That suite tests stage 1: component,
architecture, and contract *identification*. This suite tests stage 2: full
ODCS-completeness questions for exactly the contracts stage 1 hands it. Stage 2 is fixed
by each case's own `contracts.json`, rather than by a live stage-1 run. This follows the
same self-contained, hand-crafted-fixture approach the rest of this repo's golden sets
already use.

Two independent things happen for each case. Only one of them can make the test fail:

  - **Free, deterministic checks** (`_check_structural_quality` and
    `_check_contract_targets_are_grounded`). These are mechanical integrity checks on the
    collection itself: empty output, duplicate ids, duplicate or malformed question text,
    or a question aimed at a contract that is not in `contracts.json`. These checks cost
    nothing to run. `passed` in `result.json` reports this outcome, and this is what the
    test actually asserts on.
  - **The Critic.** This is a second, independent LLM call. It scores from 0 to 100 how
    completely the drafted questions would let every listed contract reach a full ODCS
    v3.0.0 specification, if a human answered every one of them. It also writes a
    one-paragraph `reason`. Its prompt lives in this directory's own `critic_prompt.jinja`.
    `score` and `reason` are recorded for information only. A probabilistic judge deciding
    pass or fail is exactly what made this pattern's predecessor flaky before (see
    `agents/stages/architecture_questions/testing/README.md`). So only the deterministic checks above decide
    pass or fail here too.

This suite is deliberately excluded from the default `pytest`/`make test` run (see
`testpaths = ["tests"]` in pyproject.toml). It calls the real LLM twice per case, once for
the Actor and once for the Critic, on purpose.

Run:
    uv run pytest agents/stages/data_contract_questions/testing/ -v
"""

import json
from pathlib import Path

import jinja2
import litellm
import pytest
from pydantic import BaseModel

from agents.shared import QuestionItem
from agents.stages.data_contract_questions.service import (
    MIN_QUESTIONS_PER_CONTRACT,
    generate_data_contract_questions_for_batch,
)
from agents.template import load_json_response
from app.config import settings
from llm import router

from agents.stages.data_contract_questions.testing.conftest import record_result

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
    """See the identical fixture in `agents/stages/architecture_questions/testing/test_golden_set.py` for the
    full reasoning. This fixture wraps the real `litellm.acompletion` call. It meters every
    call this test case makes, for both the Actor and the Critic. It is function-scoped, so
    each parametrized case gets its own fresh counters."""
    usage = {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
    original_acompletion = litellm.acompletion

    async def metered_acompletion(*args, **kwargs):
        response = await original_acompletion(*args, **kwargs)
        if getattr(response, "usage", None) is not None:
            usage["input_tokens"] += response.usage.prompt_tokens
            usage["output_tokens"] += response.usage.completion_tokens
        try:
            usage["cost_usd"] += litellm.completion_cost(completion_response=response)
        except Exception:  # noqa: BLE001 - pricing lookup can fail for an unlisted model
            pass
        return response

    monkeypatch.setattr(litellm, "acompletion", metered_acompletion)
    return usage


class EvaluationResult(BaseModel):
    score: int
    rationale: str


def _check_structural_quality(questions: list[QuestionItem], contract_count: int) -> list[str]:
    """These are deterministic integrity checks. They do not use the LLM. They only catch
    mechanical breakage, mirroring `agents.stages.architecture_questions.testing`'s own structural check.
    They also add a final gate on the coverage floor
    (`MIN_QUESTIONS_PER_CONTRACT` per contract) that
    `generate_data_contract_questions_for_batch`'s own shallow retry already aims for, in
    case those retries did not clear it."""
    violations = []
    floor = MIN_QUESTIONS_PER_CONTRACT * contract_count
    if len(questions) < floor:
        violations.append(f"only {len(questions)} question(s) drafted for {contract_count} contract(s) (< {floor})")

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


def _check_contract_targets_are_grounded(questions: list[QuestionItem], contracts: list[dict]) -> list[str]:
    """This check is deterministic and does not use the LLM. Every question's `target`
    must match a contract name this stage was actually given. This stage never discovers a
    contract on its own (see `data_contract_questions.jinja`'s PHASE 12 — NO
    INVENTION)."""
    known = {c["name"].lower() for c in contracts}
    ungrounded = sorted({q.target for q in questions if q.target.lower() not in known})
    if ungrounded:
        return [f"{len(ungrounded)} question(s) target a contract not in contracts.json: {ungrounded}"]
    return []


def _golden_set_cases() -> list[tuple[str, str, list[dict]]]:
    """Return [(name, transcript_text, contracts), ...], sorted by directory name."""
    cases = []
    for case_dir in sorted(p for p in GOLDEN_SET_DIR.iterdir() if p.is_dir()):
        transcript = (case_dir / "transcript.txt").read_text(encoding="utf-8")
        contracts = json.loads((case_dir / "contracts.json").read_text(encoding="utf-8"))
        cases.append((case_dir.name, transcript, contracts))
    return cases


def _bronze_documents_for(name: str, transcript: str) -> list[dict]:
    return [{"source_component": f"{name}.en.vtt", "content": transcript}]


def _contracts_block(contracts: list[dict]) -> str:
    return "\n".join(f"- {c['name']}: {c['producer']} -> {c['consumer']} ({c['action']})" for c in contracts)


def _questions_block(questions: list[QuestionItem]) -> str:
    return "\n".join(f"- [{q.target}] {q.question}" for q in questions) or "(no questions were drafted)"


def _load_critic_prompt_template() -> str:
    return _CRITIC_PROMPT_PATH.read_text(encoding="utf-8")


def _build_evaluation_prompt(transcript: str, contracts: list[dict], questions: list[QuestionItem]) -> list[dict]:
    template = jinja2.Template(_load_critic_prompt_template())
    content = template.render(
        transcript=transcript,
        contracts=_contracts_block(contracts),
        questions=_questions_block(questions),
    )
    return [{"role": "user", "content": content}]


async def _evaluate(transcript: str, contracts: list[dict], questions: list[QuestionItem]) -> EvaluationResult:
    messages = _build_evaluation_prompt(transcript, contracts, questions)
    response = await router.complete(
        messages,
        providers=CRITIC_PROVIDER_ORDER,
        response_format=EvaluationResult,
        temperature=0,
        # See the identical comment in agents/service.py. This setting is required for a
        # reasoning-locked OpenAI model to accept temperature=0. Anthropic's
        # reasoning-locked models have no equivalent setting. So
        # CRITIC_PROVIDER_ORDER's Anthropic-first attempt now fails every time. It falls
        # through to OpenAI instead, which ends up on the same provider as the Actor it is
        # judging.
        reasoning_effort="none",
    )
    return EvaluationResult.model_validate(load_json_response(response.choices[0].message.content))


@pytest.mark.parametrize(
    "name,transcript,contracts", _golden_set_cases(), ids=[c[0] for c in _golden_set_cases()]
)
async def test_golden_set_data_contract_question_collection(
    name: str, transcript: str, contracts: list[dict], _track_llm_cost: dict
) -> None:
    # This is the Actor step. It is the exact function `generate_data_contract_questions`
    # (the graph node) and `make questions` both call. It also writes
    # ingestion_date=<date>/data_contract_questions/<name>.json under this directory's own
    # output/ folder. (conftest.py redirects settings.output_dir there.)
    ingestion_date = f"golden-{name}"
    result = await generate_data_contract_questions_for_batch(
        ingestion_date, _bronze_documents_for(name, transcript), contracts
    )

    violations = _check_structural_quality(result.questions, len(contracts))
    violations += _check_contract_targets_are_grounded(result.questions, contracts)

    if violations:
        # Broken or near-empty output is not worth spending a Critic call to score.
        score, reason = 0, "Structural check failed: " + "; ".join(violations)
    else:
        evaluation = await _evaluate(transcript, contracts, result.questions)
        score, reason = evaluation.score, evaluation.rationale

    record_result(
        name,
        len(result.questions),
        len(contracts),
        not violations,
        score,
        reason,
        _track_llm_cost["input_tokens"],
        _track_llm_cost["output_tokens"],
        round(_track_llm_cost["cost_usd"] * USD_TO_EUR, 6),
    )

    assert not violations, f"{name}: structural check failed: {violations}"
