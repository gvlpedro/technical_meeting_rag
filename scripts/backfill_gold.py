#!/usr/bin/env python3
"""Replays Gold extraction over existing `silver_documents` rows. It runs standalone,
not through Silver's own graph.

It calls the same function the Gold nodes in `agents/graph.py` call:
`agents.stages.gold.service.extract_and_persist_gold_facts`. One implementation serves
both paths.

Run this after a change to the extraction prompt or model, to reprocess history. It also
backfills Gold for old `silver_documents` rows from before Gold existed.

Usage:
    uv run python3 scripts/backfill_gold.py                          # every silver_documents row
    uv run python3 scripts/backfill_gold.py --source-component meeting.en.vtt
    uv run python3 scripts/backfill_gold.py --force                  # re-extract even if
                                                                       # already_extracted is
                                                                       # true — use after
                                                                       # changing the extraction
                                                                       # prompt itself
"""

import argparse
import asyncio

from sqlalchemy import select

from agents.stages import gold
from db.models import SilverDocument
from db.session import async_session_factory


async def main(source_component: str | None, force: bool) -> None:
    async with async_session_factory() as session:
        # Order by ingestion_date, not by insertion order. `persist_entity_version` closes a
        # superseded row's `valid_to` using the next row's own `ingestion_date`. Processing an
        # older meeting after a newer one would close `valid_to` before the row's own date.
        query = select(SilverDocument).order_by(SilverDocument.ingestion_date, SilverDocument.version)
        if source_component:
            query = query.where(SilverDocument.source_component == source_component)
        docs = (await session.execute(query)).scalars().all()

        processed = 0
        for doc in docs:
            extracted = await gold.extract_and_persist_gold_facts(
                session,
                doc.content,
                doc.source_component,
                doc.version,
                doc.ingestion_date,
                tenant=doc.tenant,
                force=force,
            )
            if extracted:
                processed += 1
                await session.commit()
        print(f"Backfilled Gold for {processed}/{len(docs)} silver_documents row(s).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-component", default=None, help="Backfill only this source_component.")
    parser.add_argument(
        "--force", action="store_true", help="Re-extract even if this (source, version) was already processed."
    )
    args = parser.parse_args()
    asyncio.run(main(args.source_component, args.force))
