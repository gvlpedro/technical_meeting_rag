"""Records one result per golden-set question — entity-hit, whether the answer carried at
least one verified citation, and each RAGAS score (or `None` if that metric failed/timed out —
see `_score_ragas_metric` in `test_chat_eval.py` for why a single flaky judge call must never
sink the whole suite).

At the end of the run, this writes every record plus the averaged summary to
`agents/stages/gold/testing/chat_eval/output/report.json`. This mirrors the parent
`agents/stages/gold/testing/conftest.py`'s own `record_result`/`pytest_sessionfinish` pattern,
kept as a separate, independent copy here (not imported from the parent) because the report
shape is different — per-question precision and RAGAS scores, not a pass/fail step record."""

import json
from pathlib import Path

import pytest

OUTPUT_DIR = Path(__file__).parent / "output"

_results: list[dict] = []


def record_result(**fields) -> None:
    _results.append(fields)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    if not _results:
        return
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def _average(key: str) -> float | None:
        values = [r[key] for r in _results if r.get(key) is not None]
        return sum(values) / len(values) if values else None

    summary = {
        "question_count": len(_results),
        "entity_hit_rate": _average("entity_hit"),
        "citation_hit_rate": _average("has_citation"),
        "faithfulness": _average("faithfulness"),
        "answer_relevancy": _average("answer_relevancy"),
        "context_precision": _average("context_precision"),
        "context_recall": _average("context_recall"),
    }
    report = {"summary": summary, "questions": _results}
    (OUTPUT_DIR / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def pytest_terminal_summary(terminalreporter: pytest.TerminalReporter) -> None:
    report_path = OUTPUT_DIR / "report.json"
    if not report_path.exists():
        return
    terminalreporter.section("agents/stages/gold/testing/chat_eval/output/report.json — summary")
    terminalreporter.write_line(json.dumps(json.loads(report_path.read_text())["summary"], indent=2))
