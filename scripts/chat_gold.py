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
  - **Hybrid top-k retrieval**, for everything else: cosine-similarity search over
    `gold_evolution.embedding` (HNSW-indexed, `vector_cosine_ops`) fused, via Reciprocal Rank
    Fusion, with a lexical `ts_rank` search over `gold_evolution.search_vector` (a generated
    `tsvector`, GIN-indexed — see `.tmp/advanced_techniques.md` §1 and migration
    `188c1b98dd96`). The lexical half catches an exact component name, acronym, or ODCS field
    name that the embedding alone can blur. A real LLM then writes an answer from the fused
    rows.

All of this retrieval code (`find_entity_by_name_in_text`, `is_evolution_question`,
`entity_history`, `answer_evolution_question`, `embed_question`, `top_k_gold_evolution`,
`latest_versions`, `answer_question`) lives in `agents/stages/gold/service.py`. The
`agents/stages/gold/testing/` test suite uses the same top-k code, through a thin re-export in
`agents/stages/gold/testing/retrieval.py`. So there is one definition of each retrieval
strategy, not a separate copy for each caller.

`agents/stages/gold/service.py` adds three refinements on top of plain top-k retrieval. First,
`max_distance` drops rows that are too far from the question, instead of always answering
from the k closest rows regardless of how relevant they are (`--max-distance`, which defaults
to `DEFAULT_MAX_DISTANCE`; this default is not yet tuned against real usage, see its
docstring). Second, `latest_versions` tags each retrieved row as current or superseded. This
way, a question about today's state does not get answered from an old version that happens to
rank close by embedding similarity. Third, `--rerank` (opt-in, off by default) repunctuates
the deduped candidates with a real local cross-encoder before cutting to `k` — see
`top_k_gold_evolution`'s own docstring and `.tmp/advanced_techniques.md` §3. This flag is what
actually makes `rerank` reachable from this CLI at all: `top_k_gold_evolution`'s own
`rerank` parameter has no effect unless some real caller passes `rerank=True`, and before
this flag existed, nothing in this codebase ever did.

Fourth, this REPL keeps a running `history` of `(role, content)` turns across the session and
threads it through both retrieval and generation the same way `app/routers/frontend.py::chat`
does: a contextualized question (recent history + the current question) drives entity
matching/embedding/lexical search, so a follow-up like "and who approved it?" still resolves
to the right rows, while the raw history is passed to `answer_question`/
`answer_evolution_question` for reference resolution only — see
`.tmp/advanced_techniques.md` §8.

Usage:
    uv run python3 scripts/chat_gold.py
    uv run python3 scripts/chat_gold.py --k 10 --max-distance 0.7
    uv run python3 scripts/chat_gold.py --rerank
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
    """Same purpose as `app/routers/frontend.py::_contextualize_question` — folds the last
    `MAX_HISTORY_MESSAGES` turns into the text handed to retrieval, so a follow-up question
    still targets the right rows. Only used for retrieval; generation gets the raw `history`
    and the original `question` separately."""
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
    history: list[tuple[str, str]],
) -> str:
    contextualized_question = _contextualize(question, history)

    if is_evolution_question(contextualized_question):
        match = await find_entity_by_name_in_text(session, contextualized_question, tenant=tenant)
        if match is not None:
            entity_type, entity_id, matched_alias = match
            rows = await entity_history(session, entity_type, entity_id, tenant=tenant)
            return await answer_evolution_question(question, matched_alias, rows, history=history)

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
    )
    latest = await latest_versions(session, rows)
    return await answer_question(session, question, rows, latest, history=history)


async def _chat(k: int, max_distance: float | None, tenant: str, rerank: bool) -> None:
    print(f"Gold RAG chat (tenant={tenant!r}) — ask about the architecture's evolution. Ctrl+D or 'exit' to quit.\n")
    if rerank:
        print("(reranking enabled — each answer costs one extra local cross-encoder pass)\n")
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
            answer = await _answer(session, question, k, max_distance, tenant, rerank, history)
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
    args = parser.parse_args()
    asyncio.run(_chat(args.k, args.max_distance if args.max_distance >= 0 else None, args.tenant, args.rerank))
