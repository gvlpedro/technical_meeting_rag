#!/usr/bin/env python3
"""This script wipes every Bronze, Silver, and Gold row, every logged LLM call, and every
LangGraph checkpoint, from the dev database. It also wipes the `output/` directory's generated
audit files. This gives a full reset back to "nothing uploaded yet". Use it when you want to
start a fresh round of uploads, with no old transcripts, ADRs, Gold facts, or Monitor-tab call
history left over. This is the `make clean` entry point.

This script uses TRUNCATE, not DELETE, on the same list of tables that
`agents/stages/gold/testing/test_golden_set.py` already truncates between its own steps
(`_TABLES_TO_TRUNCATE` there), plus `llm_costs` — that other list does not include it either;
both are a hand-kept "every table this app writes rows to" enumeration, so a new table only
ends up wiped here once someone explicitly adds it, not automatically. `gold_aliases`,
`gold_evolution`, `silver_chunks`, `silver_clarifications`, `silver_documents`,
`bronze_documents`, `llm_costs`. The script reads this list from each model's own
`__tablename__`, so a table already listed here cannot drift out of sync with `db/models.py` —
it just does not catch a table that was never added to the list in the first place, which is
exactly what happened to `llm_costs` when it was introduced.

This script also truncates the LangGraph checkpointer's three data tables: `checkpoints`,
`checkpoint_blobs`, and `checkpoint_writes`. This makes sure no paused, half-answered
clarification thread survives the reset. It deliberately does NOT touch
`checkpoint_migrations`. That table is the checkpointer library's own schema-version
bookkeeping, not this app's data.

This script never touches `alembic_version` or the table structure. It resets DATA, not
schema. It always runs against `settings.database_url`, which is the dev database on
`localhost:5433`. Every test target in the Makefile explicitly overrides `DATABASE_URL` to
point at a separate test database instead, so this script can never reach the test database by
accident.

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
