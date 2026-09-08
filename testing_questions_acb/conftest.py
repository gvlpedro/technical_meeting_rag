"""Redirects `generate_questions_for_batch`'s `output/` writes to this directory's own
`output/` (not the shared repo-root `output/`), and collects each golden-set case's
`{name, question_count, passed, score, reason, input_tokens, output_tokens,
estimated_euro_cost}` — a summary, not the full drafted content — to write as
`testing_questions_acb/output/result.json` once the whole run finishes. The full
`mentioned_components`/`questions` for a case already live in its own
`output/ingestion_date=golden-<name>/questions/<name>.json` (written by
`generate_questions_for_batch` itself); `result.json` isn't a second copy of that, it's
the at-a-glance table across the whole golden set. Its content is then printed via
pytest's own terminal-summary hook so it's visible without needing `-s`.
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
    terminalreporter.section("testing_questions_acb/output/result.json")
    terminalreporter.write_line(result_path.read_text(encoding="utf-8"))
