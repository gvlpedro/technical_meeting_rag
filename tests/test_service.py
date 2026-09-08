import json
from pathlib import Path
from types import SimpleNamespace

import litellm
import pytest

from agents.service import generate_questions_for_batch
from app.config import settings

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _fake_response(content: str) -> SimpleNamespace:
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice], model="fake")


def _question(id_: str, question: str) -> dict:
    return {"id": id_, "scope": "component", "target": "checkout service", "requirement": id_, "question": question}


async def test_generate_questions_for_batch_returns_the_drafted_result(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "output_dir", str(tmp_path))

    drafted = [_question("component.checkout_service.status", "Is the checkout service new or unchanged?")]
    mentioned = [{"name": "checkout service", "status": "unknown"}]

    async def fake_acompletion(*, model, api_key, messages, **kwargs):
        return _fake_response(json.dumps({"mentioned_components": mentioned, "questions": drafted}))

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    bronze_documents = [{"source_component": "meeting.en.vtt", "content": "We discussed checkout."}]
    result = await generate_questions_for_batch("20260906", bronze_documents)

    assert [q.model_dump() for q in result.questions] == drafted
    assert [c.model_dump() for c in result.mentioned_components] == mentioned


async def test_generate_questions_for_batch_writes_one_file_per_distinct_source(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "output_dir", str(tmp_path))

    drafted = [_question("component.checkout_service.purpose", "What does the checkout service do?")]
    mentioned = [{"name": "checkout service", "status": "unknown"}]

    async def fake_acompletion(*, model, api_key, messages, **kwargs):
        return _fake_response(json.dumps({"mentioned_components": mentioned, "questions": drafted}))

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    bronze_documents = [
        {"source_component": "meeting_a.en.vtt", "content": "Chunk one."},
        {"source_component": "meeting_a.en.vtt", "content": "Chunk two."},  # same source, two chunks
        {"source_component": "meeting_b.es.vtt", "content": "Otro chunk."},
    ]
    await generate_questions_for_batch("20260906", bronze_documents)

    questions_dir = tmp_path / "ingestion_date=20260906" / "questions"
    written = sorted(p.name for p in questions_dir.glob("*.json"))
    assert written == ["meeting_a.json", "meeting_b.json"]  # one per distinct source, not per chunk

    for name in written:
        assert json.loads((questions_dir / name).read_text()) == {
            "mentioned_components": mentioned,
            "questions": drafted,
        }


async def test_generate_questions_for_batch_sends_the_pooled_transcript_to_the_prompt(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "output_dir", str(tmp_path))
    seen_prompts: list[str] = []

    async def fake_acompletion(*, model, api_key, messages, **kwargs):
        seen_prompts.append(messages[0]["content"])
        return _fake_response(json.dumps({"mentioned_components": [], "questions": []}))

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    bronze_documents = [
        {"source_component": "meeting.en.vtt", "content": "First half of the transcript."},
        {"source_component": "meeting.en.vtt", "content": "Second half of the transcript."},
    ]
    await generate_questions_for_batch("20260906", bronze_documents)

    assert len(seen_prompts) == 1
    assert "First half of the transcript." in seen_prompts[0]
    assert "Second half of the transcript." in seen_prompts[0]
