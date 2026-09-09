#!/usr/bin/env python3
"""Run the Silver clarification loop for one ingestion_date — `make clarify`.

Usage:
    uv run python3 scripts/clarify.py --ingestion-date 20260906
    uv run python3 scripts/clarify.py --ingestion-date 20260906 --term

    make clarify DATE=20260906                  # same as the first line above
    make clarify DATE=20260906 INTERACTIVE=1     # same as the second line above —
                                                  # the Makefile already appends --term
                                                  # for you; don't also pass it by hand.

Runs `agents.graph`'s clarification loop directly against the database — no server
needs to be running (Task 8's HTTP surface doesn't exist yet). Streams the graph node
by node (`graph.astream(..., stream_mode="updates")`), printing `→ <node_name>` as
each one runs, so the flow is visible in the terminal instead of silent until the end.
Without `--term`, an interrupt just prints the (already capped, see
`agents.graph.MAX_PENDING_QUESTIONS`) pending questions and exits — nothing to answer
them with yet. With `--term`, each pending question is asked right here with
`input()`, and the graph resumes with the answers; this can happen more than once (a
classify-stage gap, then later a Boss escalation over a contradiction — see
`doc/silver_process.md` §3). On completion — whether or not any question was asked,
since a well-covered transcript can clear classify and critic with zero interrupts —
prints every resulting ADR's full content to the terminal. `agents.graph.write_document`
also writes each one to `output/ingestion_date=<date>/adr/<transcription>.md` for
manual inspection, independent of this script.
"""

import argparse
import asyncio
import sys

import logfire
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, StateSnapshot

from agents.graph import build_graph, checkpointer_dsn
from agents.service import NoBronzeDocumentsError, transcription_base_name
from agents.state import initial_state
from app.config import settings

# agents.graph's own logfire.configure() (above) prints one console line per node —
# genuinely useful under `pytest -s`, but here it's an OpenTelemetry span exporter
# running on its own async schedule, not synchronized with this script's print()/
# input() calls: those span lines can flush interleaved with (or right on top of) the
# "> " prompt, making a script that's correctly waiting for an answer look frozen or
# broken. Reconfigure with the console off for this entry point specifically — tracing
# itself (if LOGFIRE_TOKEN is set) is unaffected, only the local stdout printer is. The
# node-by-node progress this script prints instead (see `_stream_and_print`) comes from
# `graph.astream`, not from logfire, so it stays correctly ordered around `input()`.
logfire.configure(
    token=settings.logfire_token,
    send_to_logfire="if-token-present",
    service_name="silver-clarification-loop",
    console=False,
)


async def _stream_and_print(graph: CompiledStateGraph, payload, config: dict) -> None:
    """Advances the graph one LangGraph step at a time, printing `→ <node_name>` as each
    one actually runs — this is "the flow" made visible in the terminal, synchronously,
    so it never races with a later `input()` call. The merged final/paused state itself
    is read back separately via `graph.aget_state` after this returns, not from what
    this function yields — `stream_mode="updates"` gives per-node diffs, not the merged
    state `ainvoke` would have returned."""
    async for chunk in graph.astream(payload, config=config, stream_mode="updates"):
        for node_name in chunk:
            if node_name == "__interrupt__":
                continue  # surfaced separately via graph.aget_state below
            print(f"  → {node_name}")


def _pending_interrupts(snapshot: StateSnapshot) -> list:
    return [i for task in snapshot.tasks for i in task.interrupts]


async def run(ingestion_date: str, interactive: bool) -> int:
    config = {"configurable": {"thread_id": ingestion_date}}

    async with AsyncPostgresSaver.from_conn_string(checkpointer_dsn()) as saver:
        await saver.setup()
        graph = build_graph(saver)

        # A previous invocation of this script may have already paused here (each
        # `make clarify` run is a separate process — the checkpoint is what survives
        # between them). Resume that instead of silently restarting from scratch,
        # which would re-run the whole graph (and its LLM calls) from load_bronze.
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
