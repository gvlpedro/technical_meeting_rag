"""This is a real-LLM golden-set suite for `prompts/adr_generation/generator.jinja`.

The Actor drafts a complete ADR (in the shape of `doc/adr_example.md`) from a transcript
plus its already-resolved clarifications. There are three cases, in order of increasing
difficulty:

  - `01_twitter_real_time_delivery` — uses the real
    `real_time_delivery_architecture_at_twitter.en.vtt` transcript, with a 36-question
    factual verification pass that describes how the CURRENT system works. None of the
    clarifications confirm a lifecycle status for any component. So the strictest
    expectation is that the ADR/Affected-Components/Data-Contract sections collapse to
    their "not covered" notes. There should be no decision to document, and no invention.
  - `02_checkout_loyalty_integration` — synthetic. Everything resolves cleanly. This case
    exercises the full document structure, including a populated ODCS section.
  - `03_multi_region_notification_platform` — synthetic. It combines several
    partial-information patterns at once: an out-of-scope component, an unanswered detail
    for an otherwise-in-scope component, and a data contract whose existence is confirmed
    but whose schema is not.

Two independent things happen for each case. This mirrors the same split used in
`agents.stages.architecture_questions.testing` (formerly `testing_questions_acb`):

  - **Free, deterministic checks.** These check that no placeholder markers are left in
    the document, that every `expected_components` name is actually discussed, and that no
    `excluded_components` name is tagged with a change status in the Affected Components
    table. These checks cost nothing to run. `passed`, and the test's own assertion, are
    based only on these checks.
  - **The Critic.** This is a second, independent LLM call. It scores the four general
    criteria the ADR generator was actually asked for: no placeholders, no unknown points
    for any component, a clear implementation description, and only established components
    discussed. It also scores every item in that case's own `checklist.txt` (see
    `golden_set/<name>/`). This checklist is a case-specific, mostly-deterministic
    ground-truth list built directly from that case's own `clarifications.json`. For
    example: "these exact four components, these exact statuses, zero data contracts" for
    case 01, or "this exact contract, this exact version bump, no contract for
    LoyaltyService" for case 02. The Critic reports each checklist item on its own
    (`checklist_results`), not just one overall verdict. This is recorded for information
    only. A probabilistic judge deciding pass or fail is exactly what made
    `agents.stages.architecture_questions.testing` (formerly `testing_questions_acb`) flaky before (see its
    own README). So only the deterministic checks above decide pass or fail here too.

This suite is deliberately excluded from `pytest`/`make test` (see `testpaths = ["tests"]`
in pyproject.toml). It calls the real LLM twice per case, once for the Actor and once for
the Critic, on purpose.

Run:
    uv run pytest agents/stages/adr_generation/testing/ -v
"""

import json
import re
from pathlib import Path

import jinja2
import litellm
import pytest
from pydantic import BaseModel

from agents.stages.adr_generation.prompts import QaPair, build_adr_generation_prompt
from agents.template import load_json_response
from app.config import settings
from ingestion.bronze_documents_chunker import parse_vtt
from llm import router

from agents.stages.adr_generation.testing.conftest import record_result

_CRITIC_PROMPT_PATH = Path(__file__).parent / "critic_prompt.jinja"

CRITIC_PROVIDER_ORDER = list(reversed(settings.llm_fallback_order))
USD_TO_EUR = 0.92

GOLDEN_SET_DIR = Path(__file__).parent / "golden_set"

_PLACEHOLDER_PATTERNS = [
    re.compile(r"<!--.*?-->", re.DOTALL),
    re.compile(r"\bTBD\b", re.IGNORECASE),
    re.compile(r"\bTODO\b", re.IGNORECASE),
    re.compile(r"\{\{.*?\}\}"),
    re.compile(r"^\s*\|\s*\|\s*$", re.MULTILINE),
    re.compile(r"\[Describe[^\]]*\]", re.IGNORECASE),
    re.compile(r"\bComponent A\b|\bComponent B\b"),
    re.compile(r"\bcontract-a\.json\b|\bcontract-b\.json\b"),
]

_STATUS_TAGS = ("**NEW**", "**MODIFIED**", "**REMOVED**", "**UNCHANGED**")

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def _track_llm_cost(monkeypatch: pytest.MonkeyPatch) -> dict:
    """See the identical fixture in `agents/stages/architecture_questions/testing/test_golden_set.py` for the
    full reasoning. This fixture wraps the real `litellm.acompletion` call. It meters every
    call this test case makes, for both the Actor and the Critic. It is function-scoped, so
    each case gets its own fresh counters."""
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


class ChecklistItemResult(BaseModel):
    # `item` repeats the checklist line word for word, minus its [DETERMINISTIC]/[JUDGMENT]
    # tag. This lets a result match back to its source line. It does not rely on ordering.
    item: str
    passed: bool
    note: str


class AdrCritiqueResult(BaseModel):
    has_placeholders: bool
    placeholder_examples: list[str]
    has_unknown_component_points: bool
    unknown_component_examples: list[str]
    implementation_is_clear: bool
    implementation_clarity_issues: list[str]
    only_established_components: bool
    unestablished_component_examples: list[str]
    checklist_results: list[ChecklistItemResult]
    passes: bool
    reason: str


def _check_no_placeholder_markers(document: str) -> list[str]:
    """This check is deterministic and does not use the LLM. The document must not
    contain any leftover template placeholder syntax. See the NO PLACEHOLDERS section in
    `adr_generator.jinja`. This check enforces that rule mechanically. It does not just
    trust that the model followed it."""
    violations = []
    for pattern in _PLACEHOLDER_PATTERNS:
        matches = pattern.findall(document)
        if matches:
            violations.append(f"{len(matches)} placeholder marker(s) matching {pattern.pattern!r}")
    return violations


def _check_expected_components_present(document: str, expected_components: list[str]) -> list[str]:
    """Every component whose status the case's clarifications actually confirmed must be
    discussed somewhere in the document. A document that silently drops a resolved
    component is not clear about what needs to be implemented. It is incomplete."""
    missing = [name for name in expected_components if name.lower() not in document.lower()]
    if missing:
        return [f"{len(missing)} expected component(s) not mentioned anywhere in the document: {missing}"]
    return []


def _check_excluded_components_not_marked_affected(document: str, excluded_components: list[str]) -> list[str]:
    """This check is deterministic and does not use the LLM. A component whose status was
    never confirmed (see each case's `metadata.json`) must never appear tagged with a
    change status. That is exactly the "unknown point" the Critic is asked to catch. A
    plain mention in prose, such as in the Context section, is fine. Only a line that also
    carries a status tag counts as a violation."""
    violations = []
    for name in excluded_components:
        for line in document.splitlines():
            if name.lower() in line.lower() and any(tag in line for tag in _STATUS_TAGS):
                violations.append(f"excluded component {name!r} appears tagged with a change status: {line.strip()!r}")
    return violations


def _load_case(case_dir: Path) -> tuple[str, list[QaPair], dict, str]:
    transcript_path = case_dir / "transcript.vtt"
    if transcript_path.exists():
        transcript = parse_vtt(transcript_path)
    else:
        transcript = (case_dir / "transcript.txt").read_text(encoding="utf-8")

    clarifications: list[QaPair] = json.loads((case_dir / "clarifications.json").read_text(encoding="utf-8"))
    metadata = json.loads((case_dir / "metadata.json").read_text(encoding="utf-8"))
    checklist = (case_dir / "checklist.txt").read_text(encoding="utf-8")
    return transcript, clarifications, metadata, checklist


def _golden_set_cases() -> list[tuple[str, Path]]:
    return [(d.name, d) for d in sorted(GOLDEN_SET_DIR.iterdir()) if d.is_dir()]


def _load_critic_prompt_template() -> str:
    """Read the raw text of `agents/stages/adr_generation/testing/critic_prompt.jinja`. This is the Critic's own
    prompt. It is kept as a real Jinja template here, not inline in this module. This way,
    you can edit and review it like any other prompt in this repo (see `prompts/*.jinja`).
    It is just scoped to this test suite instead of production."""
    return _CRITIC_PROMPT_PATH.read_text(encoding="utf-8")


def _build_critic_prompt(
    transcript: str, clarifications: list[QaPair], document: str, checklist: str
) -> list[dict]:
    clarifications_block = "\n".join(
        f"- Q: {c['question']}\n  A: {c['answer'] if c['answer'] is not None else '(not answered)'}"
        for c in clarifications
    )
    template = jinja2.Template(_load_critic_prompt_template())
    content = template.render(
        checklist=checklist,
        transcript=transcript,
        clarifications=clarifications_block,
        document=document,
    )
    return [{"role": "user", "content": content}]


async def _critique(
    transcript: str, clarifications: list[QaPair], document: str, checklist: str
) -> AdrCritiqueResult:
    messages = _build_critic_prompt(transcript, clarifications, document, checklist)
    response = await router.complete(
        messages,
        providers=CRITIC_PROVIDER_ORDER,
        response_format=AdrCritiqueResult,
        temperature=0,
        # See the identical comment in agents/stages/architecture_questions/testing/test_golden_set.py. This
        # setting is required for a reasoning-locked OpenAI model to accept temperature=0.
        # Anthropic's reasoning-locked models have no equivalent setting. So
        # CRITIC_PROVIDER_ORDER's Anthropic-first attempt now fails every time. It falls
        # through to OpenAI instead, which ends up on the same provider as the Actor it is
        # judging.
        reasoning_effort="none",
    )
    return AdrCritiqueResult.model_validate(load_json_response(response.choices[0].message.content))


@pytest.mark.parametrize("name,case_dir", _golden_set_cases(), ids=[n for n, _ in _golden_set_cases()])
async def test_golden_set_adr_generation(name: str, case_dir: Path, _track_llm_cost: dict) -> None:
    transcript, clarifications, metadata, checklist = _load_case(case_dir)

    # This is the Actor step. It renders adr_generator.jinja through
    # build_adr_generation_prompt, and asks the model for the final ADR document
    # directly, as raw Markdown. There is no response_format here. Like its production
    # counterpart build_synthesis_prompt, this prompt writes the document text itself.
    messages = build_adr_generation_prompt(transcript, clarifications)
    response = await router.complete(messages, temperature=0, reasoning_effort="none")
    document = response.choices[0].message.content.strip()

    output_dir = Path(__file__).parent / "output" / name
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "adr.md").write_text(document, encoding="utf-8")

    violations = _check_no_placeholder_markers(document)
    violations += _check_expected_components_present(document, metadata["expected_components"])
    violations += _check_excluded_components_not_marked_affected(document, metadata["excluded_components"])

    if violations:
        critic_passes, critic_reason = False, "Deterministic check failed, Critic not run: " + "; ".join(violations)
    else:
        critique = await _critique(transcript, clarifications, document, checklist)
        failed_items = [f"[{'FAIL' if not r.passed else 'ok'}] {r.item}: {r.note}" for r in critique.checklist_results if not r.passed]
        critic_passes = critique.passes
        critic_reason = critique.reason
        if failed_items:
            critic_reason += " | Checklist failures: " + " || ".join(failed_items)

    record_result(
        name,
        not violations,
        violations,
        critic_passes,
        critic_reason,
        _track_llm_cost["input_tokens"],
        _track_llm_cost["output_tokens"],
        round(_track_llm_cost["cost_usd"] * USD_TO_EUR, 6),
    )

    assert not violations, f"{name}: deterministic check failed: {violations}"
