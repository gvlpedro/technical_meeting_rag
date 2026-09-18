"""This module redirects `generate_architecture_questions_for_batch`'s `output/` writes to
this directory's own `output/` folder. It does not use the shared repo-root `output/`
folder.

It also collects a summary for each golden-set case: `{name, question_count, passed,
score, reason, input_tokens, output_tokens, estimated_euro_cost}`. This is a summary only,
not the full drafted content. At the end of the run, it writes every summary to
`agents/stages/architecture_questions/testing/output/result.json`.

The full `mentioned_components`/`questions` for each case already live in that case's own
`output/ingestion_date=golden-<name>/questions/<name>.json` file. That file is written by
`generate_architecture_questions_for_batch` itself. `result.json` is not a second copy of
that data. It is a small table you can scan at a glance across the whole golden set.

`result.json`'s content is then printed through pytest's own terminal-summary hook, so you
can see it without needing `-s`.
"""

import json
from pathlib import Path

import pytest

from app.config import settings

OUTPUT_DIR = Path(__file__).parent / "output"

_results: list[dict] = []


@pytest.fixture(autouse=True)
def _redirect_output_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "output_dir", str(OUTPUT_DIR))


def record_result(
    name: str,
    question_count: int,
    passed: bool,
    score: int,
    reason: str,
    input_tokens: int,
    output_tokens: int,
    estimated_euro_cost: float,
) -> None:
    _results.append(
        {
            "name": name,
            "question_count": question_count,
            "passed": passed,
            "score": score,
            "reason": reason,
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
    terminalreporter.section("agents/stages/architecture_questions/testing/output/result.json")
    terminalreporter.write_line(result_path.read_text(encoding="utf-8"))
