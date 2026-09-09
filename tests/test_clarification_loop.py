import json
import shutil
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import litellm
import pytest
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from sqlalchemy import delete, select

from agents.graph import build_graph, checkpointer_dsn
from agents.state import initial_state
from app.config import settings
from db.models import BronzeDocument, SilverChunk, SilverClarification, SilverDocument
from db.session import async_session_factory

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _fake_response(content: str) -> SimpleNamespace:
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice], model="fake")


_DEFAULT_QUESTION = {
    "id": "placeholder.question",
    "scope": "component",
    "target": "placeholder",
    "requirement": "placeholder",
    "question": "placeholder question?",
}


def _make_fake_acompletion(
    classification: dict,
    synthesis_queue: list[str],
    critique_queue: list[dict],
    generated_questions: list[dict] | None = None,
    mentioned_components: list[dict] | None = None,
):
    """Dispatches on which prompt was sent (generate / classify / synthesize /
    critic). `generate_questions` sends its whole prompt as a single `user` message
    (loaded verbatim from `prompting/roles/common/clarification_questions.jinja`), the
    other three still use a `system` + `user` pair — check `messages[0]["content"]`
    either way. `synthesis_queue` and `critique_queue` are consumed in call order —
    one entry per synthesize_document/critic_document invocation, in the order the
    graph actually makes them (first pass, then one more per redraft).

    `classification`'s entries are matched back to `generated_questions` by `id` (see
    `agents.graph.classify_questions`), so a test overriding one must override the
    other consistently — the default single placeholder question/classification pair
    is enough for tests that don't care about specific question text or ids."""
    questions = generated_questions if generated_questions is not None else [_DEFAULT_QUESTION]
    components = mentioned_components if mentioned_components is not None else []

    async def fake_acompletion(*, model, api_key, messages, **kwargs):
        content_in = messages[0]["content"]
        if "Senior Software Architecture Requirements Analyst" in content_in:
            content = json.dumps({"mentioned_components": components, "questions": questions})
        elif "You classify each question" in content_in:
            content = json.dumps(classification)
        elif "Architecture Decision Record" in content_in:
            content = synthesis_queue.pop(0)
        elif "You review a drafted architecture document" in content_in:
            content = json.dumps({"claims": critique_queue.pop(0)})
        else:
            raise AssertionError(f"unexpected prompt: {content_in[:80]!r}")
        return _fake_response(content)

    return fake_acompletion


async def _insert_bronze(ingestion_date: date, source_component: str, contents: list[str]) -> None:
    async with async_session_factory() as session:
        for content in contents:
            session.add(
                BronzeDocument(
                    ingestion_date=ingestion_date,
                    source_component=source_component,
                    content=content,
                    embedding=[0.0] * settings.embedding_dim,
                )
            )
        await session.commit()


async def _cleanup_date(ingestion_date: date) -> None:
    async with async_session_factory() as session:
        await session.execute(delete(SilverChunk).where(SilverChunk.ingestion_date == ingestion_date))
        await session.execute(
            delete(SilverClarification).where(SilverClarification.ingestion_date == ingestion_date)
        )
        await session.execute(delete(SilverDocument).where(SilverDocument.ingestion_date == ingestion_date))
        await session.execute(delete(BronzeDocument).where(BronzeDocument.ingestion_date == ingestion_date))
        await session.commit()

    questions_dir = (
        Path(settings.output_dir) / f"ingestion_date={ingestion_date.strftime('%Y%m%d')}"
    )
    if questions_dir.is_dir():
        shutil.rmtree(questions_dir)


async def _run(thread_id: str, payload) -> dict:
    """Opens a brand-new `AsyncPostgresSaver` connection and compiles a fresh graph
    for every call — every multi-call test below therefore already proves the
    checkpoint survives across separate connections/processes, not just in-memory
    state within one Python object."""
    async with AsyncPostgresSaver.from_conn_string(checkpointer_dsn()) as saver:
        await saver.setup()
        graph = build_graph(saver)
        config = {"configurable": {"thread_id": thread_id}}
        return await graph.ainvoke(payload, config=config)


DATE_GENERATED_QUESTIONS = date(2026, 6, 6)
DATE_CLEAN = date(2026, 6, 1)
DATE_CLASSIFY_INTERRUPT = date(2026, 6, 2)
DATE_CONTRADICTION = date(2026, 6, 3)
DATE_LOW_SEVERITY = date(2026, 6, 4)
DATE_BOUNDED_RETRY = date(2026, 6, 5)


async def test_generate_questions_output_is_what_classify_questions_actually_sees(monkeypatch):
    """generate_questions drafts per-component questions from the transcript
    (prompting/roles/common/clarification_questions.jinja); classify_questions must
    receive exactly that list, not some other/fixed one."""
    await _insert_bronze(
        DATE_GENERATED_QUESTIONS, "meeting_f.en.vtt", ["The checkout service was discussed."]
    )
    try:
        per_component_questions = [
            {
                "id": "component.checkout_service.status",
                "scope": "component",
                "target": "checkout service",
                "requirement": "status",
                "question": "Is the checkout service new, an evolution of something already known, or unchanged?",
            },
            {
                "id": "component.checkout_service.purpose",
                "scope": "component",
                "target": "checkout service",
                "requirement": "purpose",
                "question": "What does the checkout service do?",
            },
        ]
        mentioned_components = [{"name": "checkout service", "status": "unknown"}]
        seen_questions_blocks: list[str] = []

        async def fake_acompletion(*, model, api_key, messages, **kwargs):
            content_in = messages[0]["content"]
            if "Senior Software Architecture Requirements Analyst" in content_in:
                assert "The checkout service was discussed." in content_in  # got the real transcript
                assert "# Architecture Description / Evolution" in content_in  # got the real template
                return _fake_response(
                    json.dumps(
                        {"mentioned_components": mentioned_components, "questions": per_component_questions}
                    )
                )
            if "You classify each question" in content_in:
                seen_questions_blocks.append(messages[1]["content"])
                return _fake_response(json.dumps({"classifications": []}))
            if "Architecture Decision Record" in content_in:
                return _fake_response("# ADR — Checkout Service\n\nCheckout: unchanged.")
            if "You review a drafted architecture document" in content_in:
                return _fake_response(json.dumps({"claims": []}))
            raise AssertionError(f"unexpected prompt: {content_in[:80]!r}")

        monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

        result = await _run("generated-questions-1", initial_state("20260606"))
        assert "__interrupt__" not in result

        assert len(seen_questions_blocks) == 1
        for item in per_component_questions:
            assert item["question"] in seen_questions_blocks[0]

        # output/ingestion_date=20260606/questions/meeting_f.json — mirrors
        # input/transcriptions/ingestion_date=20260606/meeting_f.en.vtt's own name.
        questions_file = (
            Path(settings.output_dir) / "ingestion_date=20260606" / "questions" / "meeting_f.json"
        )
        assert questions_file.is_file()
        assert json.loads(questions_file.read_text()) == {
            "mentioned_components": mentioned_components,
            "questions": per_component_questions,
        }
    finally:
        await _cleanup_date(DATE_GENERATED_QUESTIONS)


async def test_clean_transcript_completes_with_no_interrupt_and_persists(monkeypatch):
    await _insert_bronze(DATE_CLEAN, "meeting_a.en.vtt", ["Team discussed the billing service."])
    try:
        classification = {
            "classifications": [{"id": "placeholder.question", "answer": "Billing", "status": "answered"}]
        }
        synthesized = "# Architecture Description / Evolution\n\nBilling service is new."

        monkeypatch.setattr(
            litellm,
            "acompletion",
            _make_fake_acompletion(classification, [synthesized], [[]]),
        )
        result = await _run("clean-1", initial_state("20260601"))
        assert "__interrupt__" not in result

        async with async_session_factory() as session:
            docs = (
                (await session.execute(select(SilverDocument).where(SilverDocument.ingestion_date == DATE_CLEAN)))
                .scalars()
                .all()
            )
            chunks = (
                (await session.execute(select(SilverChunk).where(SilverChunk.ingestion_date == DATE_CLEAN)))
                .scalars()
                .all()
            )
        assert len(docs) == 1
        assert docs[0].source_component == "meeting_a.en.vtt"
        assert "Billing service is new." in docs[0].content
        assert len(chunks) > 0
        assert all(c.embedding is not None for c in chunks)

        # Re-run for the same ingestion_date: upserts in place, no duplicates.
        monkeypatch.setattr(
            litellm,
            "acompletion",
            _make_fake_acompletion(classification, [synthesized], [[]]),
        )
        await _run("clean-2", initial_state("20260601"))

        async with async_session_factory() as session:
            docs_after = (
                (await session.execute(select(SilverDocument).where(SilverDocument.ingestion_date == DATE_CLEAN)))
                .scalars()
                .all()
            )
        assert len(docs_after) == 1  # upserted, not duplicated
    finally:
        await _cleanup_date(DATE_CLEAN)


async def test_classify_stage_interrupt_and_resume_persists_across_new_connection(monkeypatch):
    await _insert_bronze(DATE_CLASSIFY_INTERRUPT, "meeting_b.en.vtt", ["Someone mentioned a new service."])
    try:
        fake = _make_fake_acompletion(
            classification={
                "classifications": [
                    {"id": "component.new_service.owner", "answer": None, "status": "needs_clarification"}
                ]
            },
            synthesis_queue=["# Architecture Description / Evolution\n\nOwner: Alex."],
            critique_queue=[[]],
            generated_questions=[
                {
                    "id": "component.new_service.owner",
                    "scope": "component",
                    "target": "new service",
                    "requirement": "owner",
                    "question": "who owns this?",
                }
            ],
        )
        monkeypatch.setattr(litellm, "acompletion", fake)

        first = await _run("classify-interrupt-1", initial_state("20260602"))
        assert "__interrupt__" in first
        payload = first["__interrupt__"][0].value
        assert payload["origin"] == "classify"
        assert payload["pending_questions"] == ["who owns this?"]

        from langgraph.types import Command

        second = await _run("classify-interrupt-1", Command(resume={"who owns this?": "Alex"}))
        assert "__interrupt__" not in second

        async with async_session_factory() as session:
            audit_row = (
                (
                    await session.execute(
                        select(SilverDocument).where(SilverDocument.ingestion_date == DATE_CLASSIFY_INTERRUPT)
                    )
                )
                .scalars()
                .one()
            )
            clarification_row = (
                (
                    await session.execute(
                        select(SilverClarification).where(
                            SilverClarification.ingestion_date == DATE_CLASSIFY_INTERRUPT
                        )
                    )
                )
                .scalars()
                .one()
            )
        assert "Owner: Alex." in audit_row.content
        # Silver stores nothing on disk — the audit trail lives in silver_clarifications.
        assert clarification_row.question == "who owns this?"
        assert clarification_row.answer == "Alex"
    finally:
        await _cleanup_date(DATE_CLASSIFY_INTERRUPT)


async def test_contradiction_escalates_once_and_passes_after_human_informed_redraft(monkeypatch):
    await _insert_bronze(DATE_CONTRADICTION, "meeting_c.en.vtt", ["The auth service is being retired."])
    try:
        fake = _make_fake_acompletion(
            classification={"classifications": []},
            synthesis_queue=[
                "# Architecture Description / Evolution\n\nAuth service is deprecated.",
                "# Architecture Description / Evolution\n\nAuth service is active and maintained.",
            ],
            critique_queue=[
                [
                    {
                        "claim": "Auth service is deprecated.",
                        "supported": False,
                        "rationale": "transcript says retired, not deprecated-and-removed",
                        "severity": "material",
                    }
                ],
                [],
            ],
        )
        monkeypatch.setattr(litellm, "acompletion", fake)

        first = await _run("contradiction-1", initial_state("20260603"))
        assert "__interrupt__" in first
        payload = first["__interrupt__"][0].value
        assert payload["origin"] == "boss"
        assert len(payload["pending_questions"]) == 1
        question = payload["pending_questions"][0]
        assert "Auth service is deprecated." in question

        from langgraph.types import Command

        second = await _run(
            "contradiction-1", Command(resume={question: "It's retired, not deprecated."})
        )
        assert "__interrupt__" not in second

        async with async_session_factory() as session:
            doc = (
                (
                    await session.execute(
                        select(SilverDocument).where(SilverDocument.ingestion_date == DATE_CONTRADICTION)
                    )
                )
                .scalars()
                .one()
            )
        assert doc.content == "# Architecture Description / Evolution\n\nAuth service is active and maintained."
    finally:
        await _cleanup_date(DATE_CONTRADICTION)


async def test_low_severity_claim_is_downgraded_without_any_interrupt(monkeypatch):
    await _insert_bronze(DATE_LOW_SEVERITY, "meeting_d.en.vtt", ["The reporting pipeline was discussed."])
    try:
        fake = _make_fake_acompletion(
            classification={"classifications": []},
            synthesis_queue=[
                "# Architecture Description / Evolution\n\nReporting pipeline runs nightly."
            ],
            critique_queue=[
                [
                    {
                        "claim": "runs nightly",
                        "supported": False,
                        "rationale": "frequency not explicitly stated, just plausible",
                        "severity": "low",
                    }
                ]
            ],
        )
        monkeypatch.setattr(litellm, "acompletion", fake)

        result = await _run("low-severity-1", initial_state("20260604"))
        assert "__interrupt__" not in result

        async with async_session_factory() as session:
            doc = (
                (
                    await session.execute(
                        select(SilverDocument).where(SilverDocument.ingestion_date == DATE_LOW_SEVERITY)
                    )
                )
                .scalars()
                .one()
            )
        assert "runs nightly **[unknown — flagged by review]**" in doc.content
    finally:
        await _cleanup_date(DATE_LOW_SEVERITY)


async def test_claim_still_material_after_one_retry_proceeds_without_a_third_ask(monkeypatch):
    await _insert_bronze(DATE_BOUNDED_RETRY, "meeting_e.en.vtt", ["The payments service was mentioned."])
    try:
        fake = _make_fake_acompletion(
            classification={"classifications": []},
            synthesis_queue=[
                "# Architecture Description / Evolution\n\nPayments service handles refunds.",
                "# Architecture Description / Evolution\n\nPayments service handles refunds, still contested.",
            ],
            critique_queue=[
                [
                    {
                        "claim": "handles refunds",
                        "supported": False,
                        "rationale": "transcript never mentions refunds",
                        "severity": "material",
                    }
                ],
                [
                    {
                        "claim": "handles refunds",
                        "supported": False,
                        "rationale": "still not mentioned, even after clarification",
                        "severity": "material",
                    }
                ],
            ],
        )
        monkeypatch.setattr(litellm, "acompletion", fake)

        first = await _run("bounded-retry-1", initial_state("20260605"))
        assert "__interrupt__" in first
        question = first["__interrupt__"][0].value["pending_questions"][0]

        from langgraph.types import Command

        second = await _run("bounded-retry-1", Command(resume={question: "no sé"}))
        # Still flagged after the one allowed retry — proceeds anyway, never asks again.
        assert "__interrupt__" not in second

        async with async_session_factory() as session:
            doc = (
                (
                    await session.execute(
                        select(SilverDocument).where(SilverDocument.ingestion_date == DATE_BOUNDED_RETRY)
                    )
                )
                .scalars()
                .one()
            )
        assert "handles refunds **[unknown — flagged by review]**" in doc.content
    finally:
        await _cleanup_date(DATE_BOUNDED_RETRY)
