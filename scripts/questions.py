#!/usr/bin/env python3
"""Draft clarification questions for one ingestion_date's transcripts — `make questions`.

Usage:
    uv run python3 scripts/questions.py --ingestion-date 20260906

Runs `agents.service.generate_questions_for_batch` — the same logic
`agents.graph`'s `generate_questions` node uses — directly against the database, no
server and no LangGraph needed. Prints the drafted questions and writes the same
`output/ingestion_date=<date>/questions/<transcription>.json` file(s) the graph would.
"""

import argparse
import asyncio
import sys

from agents.service import NoBronzeDocumentsError, generate_questions_for_batch, load_bronze_rows
from db.session import async_session_factory


async def run(ingestion_date: str) -> int:
    async with async_session_factory() as session:
        try:
            bronze_documents = await load_bronze_rows(ingestion_date, session)
        except NoBronzeDocumentsError as exc:
            print(str(exc), file=sys.stderr)
            return 1

    result = await generate_questions_for_batch(ingestion_date, bronze_documents)

    components_summary = ", ".join(f"{c.name} ({c.status})" for c in result.mentioned_components)
    print(f"Mentioned {len(result.mentioned_components)} component(s): {components_summary}")
    print(f"Drafted {len(result.questions)} question(s) for ingestion_date={ingestion_date}:")
    for question in result.questions:
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
    args = parser.parse_args()

    sys.exit(asyncio.run(run(args.ingestion_date)))


if __name__ == "__main__":
    main()
