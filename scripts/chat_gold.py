#!/usr/bin/env python3
"""Interactive REPL over Gold's evolution history — top-k cosine-similarity retrieval over
`gold_evolution.embedding` (HNSW-indexed, `vector_cosine_ops`) followed by a real LLM answer
drafted only from the retrieved rows. `make chat` entrypoint.

The retrieval mechanism itself (`embed_question`/`top_k_gold_evolution`/`latest_versions`/
`answer_question`) lives in `agents/gold_service.py`, shared with `testing_gold_arch_evolution/`'s
own test suite (which asserts against it via `testing_gold_arch_evolution/retrieval.py`, a thin
re-export) — one definition of "how we search Gold by similarity," not a copy per caller.

Two refinements on top of plain top-k, both from `agents/gold_service.py`: `max_distance` drops
rows too far from the question instead of always answering from the k closest regardless of
relevance (`--max-distance`, defaults to `DEFAULT_MAX_DISTANCE` — untuned against real usage
yet, see its docstring), and `latest_versions` tags each retrieved row as current vs. superseded
so a question about today's state doesn't get answered from an old version that happened to rank
close by embedding similarity.

Usage:
    uv run python3 scripts/chat_gold.py
    uv run python3 scripts/chat_gold.py --k 10 --max-distance 0.7
"""

import argparse
import asyncio

from agents.gold_service import (
    DEFAULT_MAX_DISTANCE,
    answer_question,
    embed_question,
    latest_versions,
    top_k_gold_evolution,
)
from db.session import async_session_factory

DEFAULT_K = 8


async def _chat(k: int, max_distance: float | None, tenant: str) -> None:
    print(f"Gold RAG chat (tenant={tenant!r}) — ask about the architecture's evolution. Ctrl+D or 'exit' to quit.\n")
    async with async_session_factory() as session:
        while True:
            try:
                question = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return
            if not question or question.lower() in {"exit", "quit"}:
                return
            vector = await embed_question(question)
            rows = await top_k_gold_evolution(session, vector, k, max_distance=max_distance, tenant=tenant)
            latest = await latest_versions(session, rows)
            answer = await answer_question(question, rows, latest)
            print(f"\n{answer}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="Top-k rows to retrieve per question.")
    parser.add_argument(
        "--max-distance",
        type=float,
        default=DEFAULT_MAX_DISTANCE,
        help="Drop rows past this cosine distance (lower = stricter). Pass a negative value to disable.",
    )
    parser.add_argument("--tenant", default="default", help="Only search this tenant's Gold facts.")
    args = parser.parse_args()
    asyncio.run(_chat(args.k, args.max_distance if args.max_distance >= 0 else None, args.tenant))
