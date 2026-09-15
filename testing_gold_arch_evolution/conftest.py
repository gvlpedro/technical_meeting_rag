"""This module records one test result per step.

Each record has this shape: {name, passed, violations, details}.
The suite adds one extra record for the suite-level checklist.txt check.

At the end of the run, this module writes every record to
testing_gold_arch_evolution/output/result.json.

test_golden_set.py also writes full detail files for each step, under
output/<step-name>/. This file holds only the short summary.
"""

import json
from pathlib import Path

import pytest

OUTPUT_DIR = Path(__file__).parent / "output"

_results: list[dict] = []


def record_result(name: str, passed: bool, violations: list[str], details) -> None:
    _results.append({"name": name, "passed": passed, "violations": violations, "details": details})


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
    terminalreporter.section("testing_gold_arch_evolution/output/result.json")
    terminalreporter.write_line(result_path.read_text(encoding="utf-8"))
