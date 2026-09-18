#!/usr/bin/env python3
"""This script replays Gold extraction and reconciliation over existing `silver_documents`
rows. It does this on its own, without running Silver's own graph. This is the "fully
rebuildable" promise from `.tmp/gold_process_v5.md` §4.

The three Gold nodes in `agents/graph.py` (`extract_gold_facts`, `resolve_gold_identity`, and
`persist_gold_evolution`) call the same function this script calls:
`agents.stages.gold.service.extract_and_persist_gold_facts`. So there is one shared
implementation of this persist step, not two separate copies.

Use this script after you change the extraction prompt or model and want to reprocess history.
You can also use it to backfill Gold for `silver_documents` rows that existed before Gold did.

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
        query = select(SilverDocument)
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
