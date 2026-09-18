"""This module collects a summary for each golden-set case.

Each summary has this shape: `{name, passed, violations, critic_passes, critic_reason,
input_tokens, output_tokens, estimated_euro_cost}`. This is a summary only. It is not the
full generated ADR.

At the end of the run, this module writes every summary to
`agents/stages/adr_generation/testing/output/result.json`. The full generated document for each case is written
separately, next to this summary, at `output/<case-name>/adr.md` (see `test_golden_set.py`).
This way, `result.json` stays a small table you can scan at a glance across the whole
golden set.

This file mirrors the shape of `agents/stages/architecture_questions/testing/conftest.py`.
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
    terminalreporter.section("agents/stages/adr_generation/testing/output/result.json")
    terminalreporter.write_line(result_path.read_text(encoding="utf-8"))
