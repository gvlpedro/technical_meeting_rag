#!/usr/bin/env python3
"""This runs the Silver clarification loop for one ingestion_date. It is the `make clarify`
entry point.

Usage:
    uv run python3 scripts/clarify.py --ingestion-date 20260906
    uv run python3 scripts/clarify.py --ingestion-date 20260906 --term

    make clarify DATE=20260906                  # same as the first line above
    make clarify DATE=20260906 INTERACTIVE=1     # same as the second line above —
                                                  # the Makefile already appends --term
                                                  # for you; don't also pass it by hand.

This runs `agents.graph`'s clarification loop directly against the database. No server needs
to be running, because Task 8's HTTP surface does not exist yet. It streams the graph node by
node, using `graph.astream(..., stream_mode="updates")`, and prints `→ <node_name>` as each
node runs. This makes the flow visible in the terminal, instead of staying silent until the
end.

Without `--term`, an interrupt just prints the pending questions and exits. These questions
are already capped, see `agents.graph.MAX_PENDING_QUESTIONS`. There is nothing here yet to
answer them with. With `--term`, this script asks each pending question right here with
`input()`, and the graph resumes with the answers. This can happen more than once: for
example, a gap found in the classify stage, and then later a Boss escalation over a
contradiction. See `doc/silver_process.md` §3 for that flow.

When the run completes, this script prints every resulting ADR's full content to the
terminal. It does this whether or not any question was asked, since a well-covered transcript
can clear classify and critic with zero interrupts. `agents.graph.write_document` also writes
each ADR to `output/ingestion_date=<date>/adr/<transcription>.md` for manual inspection. That
write happens on its own, independent of this script.
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

# `agents.graph`'s own `logfire.configure()` call prints one console line per node. This is
# genuinely useful under `pytest -s`. But here it is an OpenTelemetry span exporter that runs
# on its own async schedule. It is not synchronized with this script's `print()` and `input()`
# calls. So those span lines can print interleaved with the "> " prompt, or even on top of it.
# This can make a script that is correctly waiting for an answer look frozen or broken.
#
# So we reconfigure logfire here, for this entry point only, with the console output turned
# off. Tracing itself still works if `LOGFIRE_TOKEN` is set. Only the local stdout printer is
# turned off. This script prints its own node-by-node progress instead (see
# `_stream_and_print`). That progress comes from `graph.astream`, not from logfire, so it
# always stays in the correct order around `input()`.
logfire.configure(
    token=settings.logfire_token,
    send_to_logfire="if-token-present",
    service_name="silver-clarification-loop",
    console=False,
)


async def _stream_and_print(graph: CompiledStateGraph, payload, config: dict) -> None:
    """This advances the graph one LangGraph step at a time. It prints `→ <node_name>` as
    each node actually runs. This is what makes "the flow" visible in the terminal. It
    happens synchronously, so it never races with a later `input()` call.

    The merged final or paused state is not read from what this function yields. It is read
    back separately, through `graph.aget_state`, after this function returns. That is because
    `stream_mode="updates"` gives per-node diffs, not the merged state that `ainvoke` would
    have returned."""
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

        # A previous run of this script may have already paused here. Each `make clarify`
        # run is a separate process. The checkpoint is what survives between those runs. So
        # we resume from that checkpoint, instead of silently restarting from scratch. A
        # silent restart would re-run the whole graph, and all its LLM calls, from
        # load_bronze.
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
