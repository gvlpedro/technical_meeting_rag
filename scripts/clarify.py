#!/usr/bin/env python3
"""Run the Silver clarification loop for one ingestion_date — `make clarify`.

Usage:
    uv run python3 scripts/clarify.py --ingestion-date 20260906
    uv run python3 scripts/clarify.py --ingestion-date 20260906 --term

Runs `agents.graph`'s clarification loop directly against the database — no server
needs to be running (Task 8's HTTP surface doesn't exist yet). Without `--term`, an
interrupt just prints the pending questions and exits — nothing to answer them with
yet. With `--term`, each pending question is asked right here with `input()`, and the
graph resumes with the answers; this can happen more than once (a classify-stage gap,
then later a Boss escalation over a contradiction — see `doc/silver_process.md` §3).
On completion — whether or not any question was asked, since a well-covered
transcript can clear classify and critic with zero interrupts — prints every
resulting document's full content to the terminal.
"""

import argparse
import asyncio
import sys

import logfire
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.types import Command

from agents.graph import build_graph, checkpointer_dsn
from agents.service import NoBronzeDocumentsError
from agents.state import initial_state
from app.config import settings

# agents.graph's own logfire.configure() (above) prints one console line per node —
# genuinely useful under `pytest -s`, but here it's an OpenTelemetry span exporter
# running on its own async schedule, not synchronized with this script's print()/
# input() calls: those span lines can flush interleaved with (or right on top of) the
# "> " prompt, making a script that's correctly waiting for an answer look frozen or
# broken. Reconfigure with the console off for this entry point specifically — tracing
# itself (if LOGFIRE_TOKEN is set) is unaffected, only the local stdout printer is.
logfire.configure(
    token=settings.logfire_token,
    send_to_logfire="if-token-present",
    service_name="silver-clarification-loop",
    console=False,
)


async def run(ingestion_date: str, interactive: bool) -> int:
    config = {"configurable": {"thread_id": ingestion_date}}

    async with AsyncPostgresSaver.from_conn_string(checkpointer_dsn()) as saver:
        await saver.setup()
        graph = build_graph(saver)

        # A previous invocation of this script may have already paused here (each
        # `make clarify` run is a separate process — the checkpoint is what survives
        # between them). Resume that instead of silently restarting from scratch,
        # which would re-run the whole graph (and its LLM calls) from load_bronze.
        # `snapshot.next` alone isn't enough to tell "paused on an interrupt" apart
        # from "a previous run's node raised and is pending retry" (e.g. the
        # NoBronzeDocumentsError case below) — only the former has real Interrupts.
        snapshot = await graph.aget_state(config)
        pending_interrupts = [i for task in snapshot.tasks for i in task.interrupts]
        if pending_interrupts:
            result = {"__interrupt__": pending_interrupts}
        else:
            try:
                result = await graph.ainvoke(initial_state(ingestion_date), config=config)
            except NoBronzeDocumentsError as exc:
                print(str(exc), file=sys.stderr)
                return 1

        while "__interrupt__" in result:
            payload = result["__interrupt__"][0].value
            questions: list[str] = payload["pending_questions"]

            if not interactive:
                print(f"Clarification needed for ingestion_date={ingestion_date} ({payload['origin']}):")
                for question in questions:
                    print(f"  - {question}")
                print("\nRe-run with --term to answer these in the terminal.")
                return 2

            print(f"\n--- Clarification needed ({payload['origin']}) ---")
            answers = {question: input(f"{question}\n> ").strip() for question in questions}
            result = await graph.ainvoke(Command(resume=answers), config=config)

    for source_component, content in result["documents"].items():
        print(f"\n{'=' * 80}\n{source_component}\n{'=' * 80}\n{content}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Silver clarification loop for one ingestion_date")
    parser.add_argument(
        "--ingestion-date",
        required=True,
        help="Compact YYYYMMDD partition to clarify, e.g. 20260906 (must already be ingested)",
    )
    parser.add_argument(
        "--term",
        action="store_true",
        help="Answer clarification questions interactively in the terminal instead of just "
        "reporting them and exiting",
    )
    args = parser.parse_args()

    sys.exit(asyncio.run(run(args.ingestion_date, args.term)))


if __name__ == "__main__":
    main()
