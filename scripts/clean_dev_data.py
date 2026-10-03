#!/usr/bin/env python3
"""Wipes every Bronze, Silver, and Gold row, every logged LLM call, and every LangGraph
checkpoint from the dev database. It also clears the `output/` directory. This resets
the app to "nothing uploaded yet". This is the `make clean` entry point.

It TRUNCATEs each table listed by its own model's `__tablename__`
(`bronze_documents`, `silver_documents`, `silver_chunks`, `silver_clarifications`,
`gold_evolution`, `gold_aliases`, `llm_costs`). A table only gets wiped once someone
adds it to this list.

It also truncates the checkpointer's three data tables (`checkpoints`,
`checkpoint_blobs`, `checkpoint_writes`), so no paused clarification thread survives
the reset. It does not touch `checkpoint_migrations` — that is the checkpointer
library's own schema bookkeeping.

It never touches `alembic_version` or the schema itself — this resets data, not
structure. It always runs against `settings.database_url`, the dev database on
`localhost:5433`. Every Makefile test target overrides `DATABASE_URL` to a separate
test database, so this script can never reach it by accident.

Usage:
    uv run python3 scripts/clean_dev_data.py
"""

import asyncio
import shutil
from pathlib import Path

from sqlalchemy import text

from app.config import settings
from db.models import (
    BronzeDocument,
    GoldAlias,
    GoldEvolution,
    LlmCost,
    SilverChunk,
    SilverClarification,
    SilverDocument,
)
from db.session import async_session_factory

_APP_TABLES = [
    GoldAlias.__tablename__,
    GoldEvolution.__tablename__,
    SilverChunk.__tablename__,
    SilverClarification.__tablename__,
    SilverDocument.__tablename__,
    BronzeDocument.__tablename__,
    LlmCost.__tablename__,
]
_CHECKPOINT_TABLES = ["checkpoint_writes", "checkpoint_blobs", "checkpoints"]


async def _truncate_all() -> None:
    async with async_session_factory() as session:
        await session.execute(text(f"TRUNCATE {', '.join(_APP_TABLES)}"))
        # Checkpoint tables only exist once a graph run has used the AsyncPostgresSaver at
        # least once. On a brand new database, skip this cleanly.
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
    print(
        "Done — Bronze/Silver/Gold, llm_costs, checkpoints, and output/ are all empty. "
        "Ready for a fresh upload."
    )


if __name__ == "__main__":
    asyncio.run(main())
