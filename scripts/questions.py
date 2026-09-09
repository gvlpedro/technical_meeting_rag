#!/usr/bin/env python3
"""Draft clarification questions for one ingestion_date's transcripts.

Usage:
    uv run python3 scripts/questions.py --ingestion-date 20260906                       # both stages
    uv run python3 scripts/questions.py --ingestion-date 20260906 --stage architecture   # stage 1 only
    uv run python3 scripts/questions.py --ingestion-date 20260906 --stage data-contract  # stage 2 only

    make questions DATE=20260906                    # both stages
    make questions-arch DATE=20260906                # stage 1 only
    make questions-data-contracts DATE=20260906       # stage 2 only

Runs the same question-generation stages `agents.graph`'s `generate_architecture_questions`/
`generate_data_contract_questions` nodes use — directly against the database, no server and no
LangGraph needed: `agents.service.generate_architecture_questions_for_batch` (components, ADR,
data-contract identification) and/or `agents.service.generate_data_contract_questions_for_batch`
(full ODCS-completeness questions for whatever contracts stage 1 identified). Each stage writes
its own audit file under `output/ingestion_date=<date>/` — `questions/` for stage 1,
`data_contract_questions/` for stage 2 — same as a real `make clarify` run would.

`--stage data-contract` alone doesn't re-run stage 1 (and doesn't re-pay for its LLM call) — it
reads stage 1's own `mentioned_data_contracts` back from its audit file
(`output/ingestion_date=<date>/questions/<transcription>.json`), so `--stage architecture` must
have been run for this `ingestion_date` at least once before.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from agents.service import (
    NoBronzeDocumentsError,
    distinct_sources,
    generate_architecture_questions_for_batch,
    generate_data_contract_questions_for_batch,
    load_bronze_rows,
    transcription_base_name,
)
from app.config import settings
from db.session import async_session_factory


def _load_previously_identified_contracts(ingestion_date: str, bronze_documents: list[dict]) -> list[dict]:
    """Reads `mentioned_data_contracts` back from stage 1's own audit file — pooled across the
    whole batch (`doc/silver_process.md` §2), so any one distinct source's file carries the
    same list. Lets `--stage data-contract` run standalone without re-running stage 1."""
    sources = distinct_sources(bronze_documents)
    name = transcription_base_name(sources[0]) if sources else ""
    audit_path = Path(settings.output_dir) / f"ingestion_date={ingestion_date}" / "questions" / f"{name}.json"
    if not audit_path.is_file():
        print(
            f"No architecture-stage output found at {audit_path} — run "
            f"`make questions-arch DATE={ingestion_date}` first.",
            file=sys.stderr,
        )
        sys.exit(1)
    data = json.loads(audit_path.read_text(encoding="utf-8"))
    if "mentioned_data_contracts" not in data:
        # A file at this exact path from before the architecture/data-contract split (or any
        # other stale writer) has `mentioned_components`/`questions` but not this key — a raw
        # KeyError here just points at line 59 of this file, not at what's actually wrong.
        print(
            f"{audit_path} exists but is missing `mentioned_data_contracts` — it looks like it "
            f"was written by an older version of the architecture stage, before it identified "
            f"data contracts. Re-run `make questions-arch DATE={ingestion_date}` to regenerate "
            f"it, then retry.",
            file=sys.stderr,
        )
        sys.exit(1)
    return data["mentioned_data_contracts"]


async def run(ingestion_date: str, stage: str) -> int:
    async with async_session_factory() as session:
        try:
            bronze_documents = await load_bronze_rows(ingestion_date, session)
        except NoBronzeDocumentsError as exc:
            print(str(exc), file=sys.stderr)
            return 1

    contracts: list[dict] = []

    if stage in ("architecture", "both"):
        arch_result = await generate_architecture_questions_for_batch(ingestion_date, bronze_documents)
        contracts = [c.model_dump() for c in arch_result.mentioned_data_contracts]

        components_summary = ", ".join(f"{c.name} ({c.status})" for c in arch_result.mentioned_components)
        print(f"Mentioned {len(arch_result.mentioned_components)} component(s): {components_summary}")
        contracts_summary = ", ".join(
            f"{c.name} ({c.producer} -> {c.consumer}, {c.action})" for c in arch_result.mentioned_data_contracts
        )
        print(f"Identified {len(arch_result.mentioned_data_contracts)} data contract(s): {contracts_summary}")
        print(f"\nDrafted {len(arch_result.questions)} architecture question(s):")
        for question in arch_result.questions:
            print(f"  - [{question.id}] {question.question}")

    if stage in ("data-contract", "both"):
        if stage == "data-contract":
            contracts = _load_previously_identified_contracts(ingestion_date, bronze_documents)

        contract_result = await generate_data_contract_questions_for_batch(
            ingestion_date, bronze_documents, contracts
        )
        print(f"\nDrafted {len(contract_result.questions)} data-contract question(s):")
        for question in contract_result.questions:
            print(f"  - [{question.id}] {question.question}")

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Draft clarification questions for one ingestion_date's transcripts"
    )
    parser.add_argument(
        "--ingestion-date",
        required=True,
        help="Compact YYYYMMDD partition to draft questions for, e.g. 20260906 "
        "(must already be ingested)",
    )
    parser.add_argument(
        "--stage",
        choices=["architecture", "data-contract", "both"],
        default="both",
        help="Which question-generation stage to run (default: both, back to back)",
    )
    args = parser.parse_args()

    sys.exit(asyncio.run(run(args.ingestion_date, args.stage)))


if __name__ == "__main__":
    main()
