"""This is a real-LLM golden set for `prompts/question_classifier.jinja`
(`agents.stages.classification.prompts.build_classification_prompt`, the exact prompt builder that
`agents/graph.py`'s `classify_questions` node calls). Before this suite existed, this stage
had no dedicated benchmark of any kind. Yet it makes the single decision that gates every
question this graph run will ever show to a human: `answered` silently drops a question,
while `needs_clarification`/`unknown` keep it.

Unlike this repo's other golden sets, correctness here has a real, hand-authored ground
truth for each question (`expected` in each case's `case.json`). This is not a
probabilistic Critic score recorded for information only. It is a deterministic pass/fail
gate. The whole point of this suite is to pin down cases where "is this actually stated"
has one clear right answer, even though the classifier's own criteria (state vs. imply vs.
assume) are inherently fuzzy. Each case's `description` explains the specific failure mode
it guards against. Read those before touching `question_classifier.jinja`.

This suite is deliberately excluded from `make test` (see `testpaths = ["tests"]` in
pyproject.toml). It makes one real LLM call per case.

Run:
    uv run pytest agents/stages/classification/testing/ -v
"""

import json
from pathlib import Path

import litellm
import pytest

from agents.stages.classification.prompts import build_classification_prompt
from agents.stages.classification.schemas import ClassificationResult
from agents.template import load_json_response
from llm import router

from agents.stages.classification.testing.conftest import record_result

GOLDEN_SET_DIR = Path(__file__).parent / "golden_set"

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def _track_llm_cost(monkeypatch: pytest.MonkeyPatch) -> dict:
    """See the identical fixture in `agents/stages/architecture_questions/testing/test_golden_set.py` for the
    full reasoning. This fixture wraps the real `litellm.acompletion` call to meter this
    case's one classify call."""
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


def _check_expectation(question_id: str, status: str, answer: str | None, expected: dict) -> list[str]:
    """Compare one classified question against its hand-authored `expected` entry. Return a
    list of human-readable failure strings. This list is empty if the classification
    satisfies every constraint the case declared for this question. Constraints add up, and
    each one is optional per question. So a case only asserts what it actually cares about:

    - `acceptable_statuses`: the classifier's own `status` must be one of these.
    - `required_answer_substrings_any_of`: when `status == "answered"`, its `answer` must
      contain at least one of these, case-insensitive. It does not need ALL of them, since a
      real answer's exact wording varies from run to run.
    - `forbidden_answer_substrings`: when answered, `answer` must contain NONE of these.
      This is what pins down a wrong inference, for example "unchanged" for a
      from-scratch system, even when `acceptable_statuses` allows "answered" in general.
    """
    failures = []
    acceptable = expected.get("acceptable_statuses")
    if acceptable is not None and status not in acceptable:
        failures.append(f"{question_id}: status={status!r}, expected one of {acceptable}")

    answer_lower = (answer or "").lower()
    if status == "answered":
        required_any_of = expected.get("required_answer_substrings_any_of")
        if required_any_of and not any(s.lower() in answer_lower for s in required_any_of):
            failures.append(
                f"{question_id}: answer {answer!r} contains none of the required substrings {required_any_of}"
            )
        forbidden = expected.get("forbidden_answer_substrings")
        if forbidden:
            hit = [s for s in forbidden if s.lower() in answer_lower]
            if hit:
                failures.append(f"{question_id}: answer {answer!r} contains forbidden substring(s) {hit}")
    return failures


def _case_dirs() -> list[Path]:
    return sorted(p for p in GOLDEN_SET_DIR.iterdir() if p.is_dir())


@pytest.mark.parametrize("case_dir", _case_dirs(), ids=lambda p: p.name)
async def test_golden_set_classifier(case_dir: Path, _track_llm_cost: dict) -> None:
    case = json.loads((case_dir / "case.json").read_text(encoding="utf-8"))
    transcript = (case_dir / "transcript.txt").read_text(encoding="utf-8")
    questions = case["questions"]
    expected = case["expected"]

    messages = build_classification_prompt(questions, transcript)
    # This uses the same temperature=0 plus reasoning_effort="none" combination that
    # agents.graph.classify_questions itself uses in production. See that function's
    # docstring for why: a reasoning-locked model rejects temperature=0 outright unless
    # this setting is also passed.
    response = await router.complete(
        messages, response_format=ClassificationResult, temperature=0, reasoning_effort="none"
    )
    result = ClassificationResult.model_validate(load_json_response(response.choices[0].message.content))

    by_id = {c.id: c for c in result.classifications}
    failures: list[str] = []
    for question in questions:
        qid = question["id"]
        classification = by_id.get(qid)
        if classification is None:
            failures.append(f"{qid}: missing from the classifier's response entirely")
            continue
        failures.extend(_check_expectation(qid, classification.status, classification.answer, expected[qid]))

    record_result(
        name=case_dir.name,
        passed=not failures,
        failures=failures,
        input_tokens=_track_llm_cost["input_tokens"],
        output_tokens=_track_llm_cost["output_tokens"],
        estimated_euro_cost=_track_llm_cost["cost_usd"] * 0.92,
    )

    assert not failures, f"{case_dir.name} ({case['description']}): " + "; ".join(failures)
