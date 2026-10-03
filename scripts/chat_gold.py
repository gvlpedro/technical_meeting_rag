#!/usr/bin/env python3
"""Interactive REPL over Gold's evolution history. This is the `make chat` entry point.

It picks one of two retrieval strategies per question:

  - **Full-history.** The question names a known entity AND asks for its history
    (`is_evolution_question`). This fetches every version of that entity, in order
    (`entity_history`), and an LLM narrates it (`answer_evolution_question`). It skips
    embedding search — the entity is already known, so there is nothing to rank.
  - **Hybrid top-k**, for everything else. It fuses a cosine-similarity search over
    `gold_evolution.embedding` with a lexical `ts_rank` search, then an LLM answers from
    the fused rows. The lexical half catches an exact name or acronym the embedding alone
    can miss.

All retrieval code lives in `agents/stages/gold/service.py`. The test suite re-exports
the same top-k code through `agents/stages/gold/testing/retrieval.py` — one definition,
not a copy per caller.

Three flags refine top-k retrieval. `--max-distance` drops rows too far from the
question. `latest_versions` tags each row as current or superseded, so a question about
today's state does not answer from an old version. `--rerank` (opt-in) re-scores
candidates with a local cross-encoder. `--expand` (opt-in) also searches with a few
LLM-generated reformulations of the question.

This REPL keeps a running `history` across the session. A contextualized question
(recent history plus the current question) drives retrieval, so a follow-up like "and
who approved it?" still resolves to the right rows. Generation gets the raw `history`
and the original question separately, for reference resolution only.

Usage:
    uv run python3 scripts/chat_gold.py
    uv run python3 scripts/chat_gold.py --k 10 --max-distance 0.7
    uv run python3 scripts/chat_gold.py --rerank
    uv run python3 scripts/chat_gold.py --expand
"""

import argparse
import asyncio

from agents.stages.gold.service import (
    DEFAULT_MAX_DISTANCE,
    MAX_HISTORY_MESSAGES,
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


def _contextualize(question: str, history: list[tuple[str, str]]) -> str:
    """Folds the last `MAX_HISTORY_MESSAGES` turns into the text handed to retrieval, so a
    follow-up question still targets the right rows. Used for retrieval only. Generation
    gets the raw `history` and the original `question` separately."""
    if not history:
        return question
    trimmed = history[-MAX_HISTORY_MESSAGES:]
    lines = "\n".join(f"{role.capitalize()}: {content}" for role, content in trimmed)
    return f"{lines}\n{question}"


async def _answer(
    session,
    question: str,
    k: int,
    max_distance: float | None,
    tenant: str,
    rerank: bool,
    expand: bool,
    history: list[tuple[str, str]],
) -> str:
    contextualized_question = _contextualize(question, history)

    if is_evolution_question(contextualized_question):
        match = await find_entity_by_name_in_text(session, contextualized_question, tenant=tenant)
        if match is not None:
            entity_type, entity_id, matched_alias = match
            rows = await entity_history(session, entity_type, entity_id, tenant=tenant)
            result = await answer_evolution_question(question, matched_alias, rows, history=history)
            return result.answer

    vector = await embed_question(contextualized_question)
    rows = await top_k_gold_evolution(
        session,
        vector,
        k,
        max_distance=max_distance,
        tenant=tenant,
        mode="hybrid",
        question_text=contextualized_question,
        rerank=rerank,
        expand=expand,
    )
    latest = await latest_versions(session, rows)
    result = await answer_question(session, question, rows, latest, history=history)
    return result.answer


async def _chat(k: int, max_distance: float | None, tenant: str, rerank: bool, expand: bool) -> None:
    print(f"Gold RAG chat (tenant={tenant!r}) — ask about the architecture's evolution. Ctrl+D or 'exit' to quit.\n")
    if rerank:
        print("(reranking enabled — each answer costs one extra local cross-encoder pass)\n")
    if expand:
        print("(query expansion enabled — each answer costs one extra LLM call for reformulations)\n")
    history: list[tuple[str, str]] = []
    async with async_session_factory() as session:
        while True:
            try:
                question = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return
            if not question or question.lower() in {"exit", "quit"}:
                return
            answer = await _answer(session, question, k, max_distance, tenant, rerank, expand, history)
            print(f"\n{answer}\n")
            history.append(("user", question))
            history.append(("assistant", answer))


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
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Repunctuate retrieved rows with a local cross-encoder before answering (off by default; "
        "adds one extra model pass per question).",
    )
    parser.add_argument(
        "--expand",
        action="store_true",
        help="Search with a few LLM-generated reformulations of the question too, not just the question "
        "itself (off by default; adds one extra LLM call per question).",
    )
    args = parser.parse_args()
    asyncio.run(
        _chat(args.k, args.max_distance if args.max_distance >= 0 else None, args.tenant, args.rerank, args.expand)
    )
