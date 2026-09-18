"""This module collects each golden-set case's `{name, passed, failures, input_tokens,
output_tokens, estimated_euro_cost}` into `agents/stages/classification/testing/output/result.json`. It
prints this through pytest's terminal-summary hook. This is the same convention every
other golden-set suite in this repo uses.

There is no `output_dir` redirect here. `agents.stages.classification.prompts.build_classification_prompt`
writes no audit file of its own, unlike the question-generation stages. So there is
nothing here to keep out of the shared repo-root `output/`."""

import json
from pathlib import Path

import pytest

OUTPUT_DIR = Path(__file__).parent / "output"

_results: list[dict] = []


def record_result(
    name: str,
    passed: bool,
    failures: list[str],
    input_tokens: int,
    output_tokens: int,
    estimated_euro_cost: float,
) -> None:
    _results.append(
        {
            "name": name,
            "passed": passed,
            "failures": failures,
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
    terminalreporter.section("agents/stages/classification/testing/output/result.json")
    terminalreporter.write_line(result_path.read_text(encoding="utf-8"))
