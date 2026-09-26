"""Golden-set + RAGAS baseline for Gold's chat retrieval — `.tmp/tasks2.md` task 4,
`.tmp/advanced_techniques.md` §6, extended by `.tmp/improve_timeline_questions_and_linage.md` §7
to also cover lineage/timeline questions (input/output contract direction, successors,
evolution-with-`valid_to`).

This is not a pass/fail gate. It is a measurement: for each question in
`golden_set/questions.json`, it runs the exact production retrieval path `app/routers/frontend.py`'s
`chat` endpoint uses — an evolution-shaped question (`gold.is_evolution_question`, e.g.
"checkout-service-evolution") goes through `find_entity_by_name_in_text` + `entity_history` +
`answer_evolution_question`, exactly like `/chat` does; every other question goes through
`top_k_gold_evolution(mode="hybrid")` + `answer_question` (`rerank`/`expand` both stay off here
too, see `app/routers/frontend.py`'s own comment on why) — against a small, known, self-seeded
corpus (`seed.py`), and records two independent kinds of signal per question:

  - **Entity-level precision** (not row-level — several `gold_evolution` rows can belong to the
    same entity, see `top_k_gold_evolution`'s own docstring on why dedup exists): did the
    expected entity actually come back among the retrieved rows.
  - **RAGAS scores** — `faithfulness`, `answer_relevancy`, `context_precision`,
    `context_recall` — an LLM-judged read on whether the generated answer is actually
    supported by what was retrieved, not just plausible-sounding.

It ALSO records a deterministic, non-LLM-judged citation check (`has_citation`) — whether
`answer_question`'s own verified citations (`.tmp/tasks2.md` task 1) backed the answer at all.
This is the "cross-check faithfulness against the structured citations, not just an LLM judge"
this task's own plan called for: two independent signals for the same underlying question
("is this answer actually grounded"), one mechanical and free, one an LLM's opinion.

KNOWN ISSUE, found while building this suite, not a bug in this codebase: `ragas==0.4.3`'s
`AnswerRelevancy` AND `ContextPrecision` metrics hang against this project's dependency-resolved
`langchain-community`/`instructor` versions — confirmed on a real, verified run (not a one-off):
both timed out on every one of the 4 golden-set questions, every single time, while
`faithfulness`/`context_recall` returned real scores (`1.0`) on all 4. `_score_ragas_metric`
below wraps every metric call in a bounded timeout for exactly this reason, so two external
judges reliably misbehaving never blocks the whole suite — `answer_relevancy`/`context_precision`
are expected to come back `null` in the report until a future `ragas` release fixes this
upstream, not a sign this suite itself is broken.

The RAGAS judge model is `gpt-4o-mini`, NOT `settings.openai_model`
(`app/config.py`) — this project's own configured model is a reasoning-locked model that
rejects the plain `max_tokens` parameter ragas's `instructor`-based LLM wrapper sends, and nothing
about the judge model needs to match the app's own production model anyway.

pyproject.toml excludes this suite from `make test` (see its own `testpaths` comment). Run it
on its own:

    make test-gold-chat-eval
"""

import asyncio
import json
from pathlib import Path
from uuid import uuid4

import pytest
from openai import AsyncOpenAI
from sqlalchemy import delete

from agents.stages import gold
from agents.stages.gold.schemas import GroundedAnswer
from agents.stages.gold.service import _evolution_step_line
from agents.stages.gold.testing.chat_eval.conftest import record_result
from agents.stages.gold.testing.chat_eval.seed import seed_corpus
from app.config import settings
from db.models import GoldAlias, GoldEvolution
from db.session import async_session_factory

pytestmark = pytest.mark.anyio

GOLDEN_SET_PATH = Path(__file__).parent / "golden_set" / "questions.json"
_RAGAS_JUDGE_MODEL = "gpt-4o-mini"
_RAGAS_EMBEDDING_MODEL = "text-embedding-3-small"
_METRIC_TIMEOUT_SECONDS = 60
_TOP_K = 5


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _load_golden_set() -> list[dict]:
    return json.loads(GOLDEN_SET_PATH.read_text(encoding="utf-8"))


def _build_ragas_judge():
    """One shared judge (LLM + embeddings) for the whole run — see this module's own docstring
    for why `gpt-4o-mini`, not this app's own configured model."""
    from ragas.embeddings.base import embedding_factory
    from ragas.llms import llm_factory

    client = AsyncOpenAI(api_key=settings.openai_api_key)
    llm = llm_factory(_RAGAS_JUDGE_MODEL, client=client)
    embeddings = embedding_factory("openai", _RAGAS_EMBEDDING_MODEL, client=client)
    return llm, embeddings


async def _score_ragas_metric(name: str, coro) -> float | None:
    """Runs one RAGAS metric call under a hard timeout, catching any failure — a hang (see
    `answer_relevancy`'s own known issue, this module's docstring), a rate limit, a transient
    API error. Returns `None` on any of those instead of ever letting one flaky external judge
    call sink the whole suite. `None` is a visible gap in the report to investigate, not a
    crash that hides every other question's real scores."""
    try:
        result = await asyncio.wait_for(coro, timeout=_METRIC_TIMEOUT_SECONDS)
        return float(result.value)
    except Exception as exc:  # noqa: BLE001 - one judge metric's failure must never fail the suite
        print(f"  [{name}] failed or timed out: {exc!r}")
        return None


async def _ragas_scores(llm, embeddings, *, question: str, answer: str, contexts: list[str], reference: str) -> dict:
    from ragas.metrics.collections import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness

    faithfulness = Faithfulness(llm=llm)
    answer_relevancy = AnswerRelevancy(llm=llm, embeddings=embeddings)
    context_precision = ContextPrecision(llm=llm)
    context_recall = ContextRecall(llm=llm)

    return {
        "faithfulness": await _score_ragas_metric(
            "faithfulness",
            faithfulness.ascore(user_input=question, response=answer, retrieved_contexts=contexts),
        ),
        "answer_relevancy": await _score_ragas_metric(
            "answer_relevancy", answer_relevancy.ascore(user_input=question, response=answer)
        ),
        "context_precision": await _score_ragas_metric(
            "context_precision",
            context_precision.ascore(user_input=question, reference=reference, retrieved_contexts=contexts),
        ),
        "context_recall": await _score_ragas_metric(
            "context_recall",
            context_recall.ascore(user_input=question, retrieved_contexts=contexts, reference=reference),
        ),
    }


def _entity_hit(rows, expected_entity_type: str, expected_entity_id: str) -> bool:
    return any(row.entity_type == expected_entity_type and row.entity_id == expected_entity_id for row in rows)


async def _retrieve_and_answer(session, tenant: str, case: dict, expected_entity_id: str):
    """Branches exactly the way `app/routers/frontend.py`'s `chat` endpoint does: an
    evolution-shaped question goes through the full-history path, everything else through
    ordinary hybrid retrieval. Returns `(rows, result, entity_hit, contexts)` — `contexts` is
    what actually gets scored by RAGAS below, rendered the same way `answer_question`'s own
    prompt renders it (`build_context_lines`/`_evolution_step_line`), not bare `row.narrative` —
    see `build_context_lines`'s own docstring for why that distinction matters for faithfulness
    and context_recall specifically."""
    if gold.is_evolution_question(case["question"]):
        match = await gold.find_entity_by_name_in_text(session, case["question"], tenant=tenant)
        entity_hit = match is not None and match[1] == expected_entity_id
        if match is None:
            return [], GroundedAnswer(answer="No entity found in the question."), entity_hit, []
        entity_type, entity_id, matched_alias = match
        rows = await gold.entity_history(session, entity_type, entity_id, tenant=tenant)
        result = await gold.answer_evolution_question(case["question"], matched_alias, rows)
        return rows, result, entity_hit, [_evolution_step_line(row) for row in rows]

    vector = await gold.embed_question(case["question"])
    rows = await gold.top_k_gold_evolution(
        session, vector, k=_TOP_K, tenant=tenant, mode="hybrid", question_text=case["question"],
    )
    entity_hit = _entity_hit(rows, case["expected_entity_type"], expected_entity_id)
    latest = await gold.latest_versions(session, rows)
    result = await gold.answer_question(session, case["question"], rows, latest)
    return rows, result, entity_hit, await gold.build_context_lines(session, rows, latest, case["question"])


async def test_chat_eval_baseline():
    tenant = f"chat-eval-{uuid4().hex[:8]}"
    llm, embeddings = _build_ragas_judge()

    entity_hits: dict[str, bool] = {}
    try:
        async with async_session_factory() as session:
            entity_ids = await seed_corpus(session, tenant)

        for case in _load_golden_set():
            expected_entity_id = entity_ids[case["expected_canonical_name"]]

            async with async_session_factory() as session:
                rows, result, entity_hit, contexts = await _retrieve_and_answer(
                    session, tenant, case, expected_entity_id
                )
                entity_hits[case["id"]] = entity_hit

            ragas_scores = await _ragas_scores(
                llm,
                embeddings,
                question=case["question"],
                answer=result.answer,
                contexts=contexts,
                reference=case["reference"],
            )

            record_result(
                id=case["id"],
                question=case["question"],
                expected_canonical_name=case["expected_canonical_name"],
                entity_hit=entity_hit,
                has_citation=len(result.citations) > 0,
                answer=result.answer,
                **ragas_scores,
            )

        # No fixed pass/fail threshold on the RAGAS scores — this suite exists to produce that
        # measurement (see the module docstring), not to gate on it yet. The one thing that
        # DOES gate it: retrieval must find the right entity for every golden-set question.
        # Every miss here means either the seeded corpus or the golden set itself is broken —
        # that is a bug in this suite, not a real-world retrieval-quality signal worth
        # reporting as a soft score.
        missed = [case_id for case_id, hit in entity_hits.items() if not hit]
        assert not missed, f"retrieval missed the expected entity for: {missed}"
    finally:
        async with async_session_factory() as session:
            await session.execute(delete(GoldEvolution).where(GoldEvolution.tenant == tenant))
            await session.execute(delete(GoldAlias).where(GoldAlias.tenant == tenant))
            await session.commit()
