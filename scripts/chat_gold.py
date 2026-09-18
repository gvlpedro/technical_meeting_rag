#!/usr/bin/env python3
"""This is an interactive REPL over Gold's evolution history. This is the `make chat` entry
point. It picks between two different retrieval strategies per question, not just one:

  - **Full-history retrieval.** When the question both names a known component or data
    contract (`find_entity_by_name_in_text`) AND reads as asking for its history
    (`is_evolution_question`, e.g. "how has X evolved over time?" / "qué evolución ha tenido
    X"), this fetches EVERY version of that one entity, in order (`entity_history`), and a
    real LLM narrates it chronologically (`answer_evolution_question`) — who introduced it,
    when, why, and what changed at each later step. This never uses embedding similarity: the
    entity is already known, so there is nothing to rank, and top-k could otherwise drop an
    early version whose narrative just does not word-match the question.
  - **Top-k similarity retrieval**, for everything else. This does cosine-similarity search
    over `gold_evolution.embedding` (HNSW-indexed, using `vector_cosine_ops`), then a real LLM
    writes an answer from the retrieved rows.

All of this retrieval code (`find_entity_by_name_in_text`, `is_evolution_question`,
`entity_history`, `answer_evolution_question`, `embed_question`, `top_k_gold_evolution`,
`latest_versions`, `answer_question`) lives in `agents/stages/gold/service.py`. The
`agents/stages/gold/testing/` test suite uses the same top-k code, through a thin re-export in
`agents/stages/gold/testing/retrieval.py`. So there is one definition of each retrieval
strategy, not a separate copy for each caller.

`agents/stages/gold/service.py` adds two refinements on top of plain top-k retrieval. First,
`max_distance` drops rows that are too far from the question, instead of always answering
from the k closest rows regardless of how relevant they are (`--max-distance`, which defaults
to `DEFAULT_MAX_DISTANCE`; this default is not yet tuned against real usage, see its
docstring). Second, `latest_versions` tags each retrieved row as current or superseded. This
way, a question about today's state does not get answered from an old version that happens to
rank close by embedding similarity.

Usage:
    uv run python3 scripts/chat_gold.py
    uv run python3 scripts/chat_gold.py --k 10 --max-distance 0.7
"""

import argparse
import asyncio

from agents.stages.gold.service import (
    DEFAULT_MAX_DISTANCE,
    answer_evolution_question,
    answer_question,
    embed_question,
    entity_history,
    find_entity_by_name_in_text,
    is_evolution_question,
    latest_versions,
    top_k_gold_evolution,
)
from db.session import async_session_factory

DEFAULT_K = 8


async def _answer(session, question: str, k: int, max_distance: float | None, tenant: str) -> str:
    if is_evolution_question(question):
        match = await find_entity_by_name_in_text(session, question, tenant=tenant)
        if match is not None:
            entity_type, entity_id, matched_alias = match
            rows = await entity_history(session, entity_type, entity_id, tenant=tenant)
            return await answer_evolution_question(question, matched_alias, rows)

    vector = await embed_question(question)
    rows = await top_k_gold_evolution(session, vector, k, max_distance=max_distance, tenant=tenant)
    latest = await latest_versions(session, rows)
    return await answer_question(question, rows, latest)


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
            answer = await _answer(session, question, k, max_distance, tenant)
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
