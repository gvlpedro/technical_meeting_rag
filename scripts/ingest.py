#!/usr/bin/env python3
"""Ingest one ingestion_date's transcripts into `bronze_documents` — `make ingestion`.

Usage:
    uv run python3 scripts/ingest.py --ingestion-date 20260906

Same logic `POST /v1/ingest` runs (`ingestion/service.py`), called directly against the
database instead of through the FastAPI app — no server needs to be running.
"""

import argparse
import asyncio
import sys

from db.session import async_session_factory
from ingestion.service import NoTranscriptsFoundError, ingest_bronze


async def run(ingestion_date: str) -> int:
    async with async_session_factory() as session:
        try:
            result = await ingest_bronze(ingestion_date, session)
        except ValueError:
            print(f"Invalid --ingestion-date '{ingestion_date}', expected YYYYMMDD", file=sys.stderr)
            return 1
        except NoTranscriptsFoundError as exc:
            print(str(exc), file=sys.stderr)
            return 1

    print(f"Ingested {len(result.files_ingested)} file(s) into ingestion_date={ingestion_date}:")
    for name in result.files_ingested:
        print(f"  - {name}")
    print(f"{result.chunks_created} chunk(s) created.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest one ingestion_date's transcripts into Bronze")
    parser.add_argument(
        "--ingestion-date",
        required=True,
        help="Compact YYYYMMDD partition to ingest, e.g. 20260906 "
        "(input/transcriptions/ingestion_date=20260906/)",
    )
    args = parser.parse_args()

    sys.exit(asyncio.run(run(args.ingestion_date)))


if __name__ == "__main__":
    main()
