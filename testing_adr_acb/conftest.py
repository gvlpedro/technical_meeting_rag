"""Collects each golden-set case's `{name, passed, violations, critic_passes, critic_reason,
input_tokens, output_tokens, estimated_euro_cost}` — a summary, not the full generated ADR —
into `testing_adr_acb/output/result.json` once the whole run finishes. The full generated
document for a case is written separately, alongside this summary, at
`output/<case-name>/adr.md` (see `test_golden_set.py`), so `result.json` stays a small
at-a-glance table across the whole golden set. Mirrors `testing_questions_acb/conftest.py`'s
own shape.
"""

import json
from pathlib import Path

import pytest

OUTPUT_DIR = Path(__file__).parent / "output"

_results: list[dict] = []


def record_result(
    name: str,
    passed: bool,
    violations: list[str],
    critic_passes: bool,
    critic_reason: str,
    input_tokens: int,
    output_tokens: int,
    estimated_euro_cost: float,
) -> None:
    _results.append(
        {
            "name": name,
            "passed": passed,
            "violations": violations,
            "critic_passes": critic_passes,
            "critic_reason": critic_reason,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "estimated_euro_cost": estimated_euro_cost,
        }
    )


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    if not _results:
        return
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "result.json").write_text(
        json.dumps(_results, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def pytest_terminal_summary(terminalreporter: pytest.TerminalReporter) -> None:
    result_path = OUTPUT_DIR / "result.json"
    if not result_path.exists():
        return
    terminalreporter.section("testing_adr_acb/output/result.json")
    terminalreporter.write_line(result_path.read_text(encoding="utf-8"))
