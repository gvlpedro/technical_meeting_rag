#!/usr/bin/env python3
"""Wipes every Bronze/Silver/Gold row and LangGraph checkpoint from the dev database, plus the
`output/` directory's generated audit files — a full reset back to "nothing uploaded yet",
for when you want to start a fresh round of uploads without old transcripts, ADRs, or Gold
facts still lying around. `make clean` entrypoint.

TRUNCATEs (not DELETEs) the same table list `testing_gold_arch_evolution/test_golden_set.py`
already truncates between its own steps — one definition of "every table this app writes rows
to", read from each model's own `__tablename__` so it can't drift out of sync with `db/models.py`
(`gold_aliases`, `gold_evolution`, `silver_chunks`, `silver_clarifications`, `silver_documents`,
`bronze_documents`). Also truncates the LangGraph checkpointer's own three data tables
(`checkpoints`/`checkpoint_blobs`/`checkpoint_writes`) so no paused, half-answered clarification
thread survives the reset — deliberately NOT `checkpoint_migrations`, which is the checkpointer
library's own schema-version bookkeeping, not this app's data.

Never touches `alembic_version` or table structure — this resets DATA, not schema. Always runs
against `settings.database_url` (the dev database on `localhost:5433`; every test target in the
Makefile explicitly overrides `DATABASE_URL` to the separate test database instead, so this
script can never reach it by accident).

Usage:
    uv run python3 scripts/clean_dev_data.py
"""

import asyncio
import shutil
from pathlib import Path

from sqlalchemy import text

from app.config import settings
from db.models import BronzeDocument, GoldAlias, GoldEvolution, SilverChunk, SilverClarification, SilverDocument
from db.session import async_session_factory

_APP_TABLES = [
    GoldAlias.__tablename__,
    GoldEvolution.__tablename__,
    SilverChunk.__tablename__,
    SilverClarification.__tablename__,
    SilverDocument.__tablename__,
    BronzeDocument.__tablename__,
]
_CHECKPOINT_TABLES = ["checkpoint_writes", "checkpoint_blobs", "checkpoints"]


async def _truncate_all() -> None:
    async with async_session_factory() as session:
        await session.execute(text(f"TRUNCATE {', '.join(_APP_TABLES)}"))
        # Checkpoint tables only exist once a graph run has actually used the
        # AsyncPostgresSaver at least once — skip cleanly on a brand new database.
        existing = (
            await session.execute(
                text(
                    "SELECT tablename FROM pg_tables WHERE schemaname = 'public' "
                    "AND tablename = ANY(:names)"
                ),
                {"names": _CHECKPOINT_TABLES},
            )
        ).scalars().all()
        if existing:
            await session.execute(text(f"TRUNCATE {', '.join(existing)}"))
        await session.commit()


def _reset_output_dir() -> None:
    output_dir = Path(settings.output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


async def main() -> None:
    print(f"Wiping all app data from {settings.database_url!r} and resetting {settings.output_dir!r}...")
    await _truncate_all()
    _reset_output_dir()
    print("Done — Bronze/Silver/Gold, checkpoints, and output/ are all empty. Ready for a fresh upload.")


if __name__ == "__main__":
    asyncio.run(main())
