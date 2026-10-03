#!/usr/bin/env python3
"""Runs the Silver clarification loop for one ingestion_date. This is the `make clarify`
entry point.

Usage:
    uv run python3 scripts/clarify.py --ingestion-date 20260906
    uv run python3 scripts/clarify.py --ingestion-date 20260906 --term

    make clarify DATE=20260906                  # same as the first line above
    make clarify DATE=20260906 INTERACTIVE=1     # same as the second line above —
                                                  # the Makefile already appends --term.

This calls `agents.graph`'s clarification loop directly against the database. No
server needs to run. It streams the graph node by node and prints `→ <node_name>` as
each node runs, so the flow stays visible instead of going silent until the end.

Without `--term`, an interrupt prints the pending questions and exits. With `--term`,
this script asks each pending question with `input()` and resumes the graph with the
answers. This can happen more than once in one run. See `doc/silver_process.md` §3.

When the run finishes, this script prints every resulting ADR's full content.
`agents.graph.write_document` also writes each ADR to
`output/ingestion_date=<date>/adr/<transcription>.md`, on its own.
"""

import argparse
import asyncio
import sys

import logfire
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, StateSnapshot

from agents.graph import build_graph, checkpointer_dsn
from agents.shared import NoBronzeDocumentsError, transcription_base_name
from agents.state import initial_state
from app.config import settings

# `agents.graph`'s own `logfire.configure()` prints one console line per node. That print
# runs on its own async schedule, out of sync with this script's `print()`/`input()`
# calls, so lines can interleave with the "> " prompt and make the script look frozen.
#
# This reconfigures logfire for this entry point with console output off. Tracing still
# works if `LOGFIRE_TOKEN` is set. This script prints its own progress instead (see
# `_stream_and_print`), from `graph.astream`, which stays in order around `input()`.
logfire.configure(
    token=settings.logfire_token,
    send_to_logfire="if-token-present",
    service_name="silver-clarification-loop",
    console=False,
)


async def _stream_and_print(graph: CompiledStateGraph, payload, config: dict) -> None:
    """Advances the graph one step at a time and prints `→ <node_name>` as each node runs.
    This runs synchronously, so it never races with a later `input()` call.

    Read the final or paused state back separately, with `graph.aget_state`, after this
    returns. `stream_mode="updates"` yields per-node diffs, not the merged state."""
    async for chunk in graph.astream(payload, config=config, stream_mode="updates"):
        for node_name in chunk:
            if node_name == "__interrupt__":
                continue  # We handle this separately, below, through graph.aget_state.
            print(f"  → {node_name}")


def _pending_interrupts(snapshot: StateSnapshot) -> list:
    return [i for task in snapshot.tasks for i in task.interrupts]


async def run(ingestion_date: str, interactive: bool) -> int:
    config = {"configurable": {"thread_id": ingestion_date}}

    async with AsyncPostgresSaver.from_conn_string(checkpointer_dsn()) as saver:
        await saver.setup()
        graph = build_graph(saver)

        # Each `make clarify` run is a new process. The checkpoint survives between runs,
        # so this resumes from it instead of restarting — a restart would re-run the whole
        # graph, and all its LLM calls, from load_bronze.
        snapshot = await graph.aget_state(config)
        if not _pending_interrupts(snapshot):
            print(f"--- Silver clarification loop: ingestion_date={ingestion_date} ---")
            try:
                await _stream_and_print(graph, initial_state(ingestion_date), config)
            except NoBronzeDocumentsError as exc:
                print(str(exc), file=sys.stderr)
                return 1
            snapshot = await graph.aget_state(config)

        while _pending_interrupts(snapshot):
            payload = _pending_interrupts(snapshot)[0].value
            questions: list[str] = payload["pending_questions"]

            if not interactive:
                print(f"Clarification needed for ingestion_date={ingestion_date} ({payload['origin']}):")
                for question in questions:
                    print(f"  - {question}")
                print("\nRe-run with --term to answer these in the terminal.")
                return 2

            print(f"\n--- Clarification needed ({payload['origin']}) — {len(questions)} question(s) ---")
            answers = {question: input(f"{question}\n> ").strip() for question in questions}
            await _stream_and_print(graph, Command(resume=answers), config)
            snapshot = await graph.aget_state(config)

    result = snapshot.values
    print(f"\n--- ADR(s) generated for ingestion_date={ingestion_date} ---")
    for source_component, content in result["documents"].items():
        version = result["document_versions"].get(source_component)
        print(f"\n{'=' * 80}\n{source_component} (version {version})\n{'=' * 80}\n{content}")
        base_name = transcription_base_name(source_component)
        print(f"\n(also written to output/ingestion_date={ingestion_date}/adr/{base_name}.md)")
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
