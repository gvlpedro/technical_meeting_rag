import json
from types import SimpleNamespace

import litellm
import pytest

from agents.service import (
    generate_architecture_questions_for_batch,
    generate_data_contract_questions_for_batch,
    mentions_grounded_in_source,
)
from app.config import settings

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _fake_response(content: str) -> SimpleNamespace:
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice], model="fake")


def _question(id_: str, question: str, scope: str = "component", target: str = "checkout service") -> dict:
    return {"id": id_, "scope": scope, "target": target, "requirement": id_, "question": question}


async def test_generate_architecture_questions_for_batch_returns_the_drafted_result(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "output_dir", str(tmp_path))

    drafted = [_question("component.checkout_service.status", "Is the checkout service new or unchanged?")]
    mentioned = [{"name": "checkout service", "status": "unknown"}]
    contracts = [{"name": "checkout-completed", "producer": "checkout service", "consumer": "unknown", "action": "unknown"}]

    async def fake_acompletion(*, model, api_key, messages, **kwargs):
        return _fake_response(
            json.dumps(
                {"mentioned_components": mentioned, "mentioned_data_contracts": contracts, "questions": drafted}
            )
        )

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    bronze_documents = [{"source_component": "meeting.en.vtt", "content": "We discussed checkout."}]
    result = await generate_architecture_questions_for_batch("20260906", bronze_documents)

    assert [q.model_dump() for q in result.questions] == drafted
    assert [c.model_dump() for c in result.mentioned_components] == mentioned
    assert [c.model_dump() for c in result.mentioned_data_contracts] == contracts


async def test_generate_architecture_questions_for_batch_writes_one_file_per_distinct_source(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(settings, "output_dir", str(tmp_path))

    drafted = [_question("component.checkout_service.purpose", "What does the checkout service do?")]
    mentioned = [{"name": "checkout service", "status": "unknown"}]

    async def fake_acompletion(*, model, api_key, messages, **kwargs):
        return _fake_response(
            json.dumps({"mentioned_components": mentioned, "mentioned_data_contracts": [], "questions": drafted})
        )

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    bronze_documents = [
        {"source_component": "meeting_a.en.vtt", "content": "Chunk one."},
        {"source_component": "meeting_a.en.vtt", "content": "Chunk two."},  # same source, two chunks
        {"source_component": "meeting_b.es.vtt", "content": "Otro chunk."},
    ]
    await generate_architecture_questions_for_batch("20260906", bronze_documents)

    questions_dir = tmp_path / "ingestion_date=20260906" / "questions"
    written = sorted(p.name for p in questions_dir.glob("*.json"))
    assert written == ["meeting_a.json", "meeting_b.json"]  # one per distinct source, not per chunk

    for name in written:
        assert json.loads((questions_dir / name).read_text()) == {
            "mentioned_components": mentioned,
            "mentioned_data_contracts": [],
            "questions": drafted,
        }


async def test_generate_architecture_questions_for_batch_sends_the_pooled_transcript_to_the_prompt(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(settings, "output_dir", str(tmp_path))
    seen_prompts: list[str] = []

    async def fake_acompletion(*, model, api_key, messages, **kwargs):
        seen_prompts.append(messages[0]["content"])
        return _fake_response(json.dumps({"mentioned_components": [], "mentioned_data_contracts": [], "questions": []}))

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    bronze_documents = [
        {"source_component": "meeting.en.vtt", "content": "First half of the transcript."},
        {"source_component": "meeting.en.vtt", "content": "Second half of the transcript."},
    ]
    await generate_architecture_questions_for_batch("20260906", bronze_documents)

    assert len(seen_prompts) == 1
    assert "First half of the transcript." in seen_prompts[0]
    assert "Second half of the transcript." in seen_prompts[0]


async def test_generate_data_contract_questions_for_batch_skips_the_llm_call_when_nothing_identified(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(settings, "output_dir", str(tmp_path))
    calls = 0

    async def fake_acompletion(*, model, api_key, messages, **kwargs):
        nonlocal calls
        calls += 1
        return _fake_response(json.dumps({"questions": []}))

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    bronze_documents = [{"source_component": "meeting.en.vtt", "content": "We discussed checkout."}]
    result = await generate_data_contract_questions_for_batch("20260906", bronze_documents, [])

    assert result.questions == []
    assert calls == 0  # no contracts identified -> nothing to spend an LLM call asking about


async def test_generate_data_contract_questions_for_batch_returns_the_drafted_result(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "output_dir", str(tmp_path))

    drafted = [_question("contract.checkout_completed.schema", "What fields does checkout-completed carry?", "data_contract", "checkout-completed")]
    mentioned_data_contracts = [
        {"name": "checkout-completed", "producer": "checkout service", "consumer": "order history", "action": "forward-update"}
    ]

    async def fake_acompletion(*, model, api_key, messages, **kwargs):
        return _fake_response(json.dumps({"questions": drafted}))

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    bronze_documents = [{"source_component": "meeting.en.vtt", "content": "We discussed checkout."}]
    result = await generate_data_contract_questions_for_batch(
        "20260906", bronze_documents, mentioned_data_contracts
    )

    assert [q.model_dump() for q in result.questions] == drafted


async def test_generate_data_contract_questions_for_batch_writes_its_own_audit_file(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "output_dir", str(tmp_path))

    drafted = [_question("contract.checkout_completed.schema", "What fields does checkout-completed carry?", "data_contract", "checkout-completed")]
    mentioned_data_contracts = [
        {"name": "checkout-completed", "producer": "checkout service", "consumer": "order history", "action": "forward-update"}
    ]

    async def fake_acompletion(*, model, api_key, messages, **kwargs):
        return _fake_response(json.dumps({"questions": drafted}))

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    bronze_documents = [{"source_component": "meeting.en.vtt", "content": "We discussed checkout."}]
    await generate_data_contract_questions_for_batch("20260906", bronze_documents, mentioned_data_contracts)

    written = tmp_path / "ingestion_date=20260906" / "data_contract_questions" / "meeting.json"
    assert json.loads(written.read_text()) == {"questions": drafted}


async def test_generate_data_contract_questions_for_batch_sends_the_identified_contracts_to_the_prompt(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(settings, "output_dir", str(tmp_path))
    seen_prompts: list[str] = []
    # 5 questions for 1 contract clears MIN_QUESTIONS_PER_CONTRACT — a shallower result would
    # trigger the shallow-retry loop and this test only cares about the first prompt sent.
    not_shallow = [
        _question(f"contract.checkout_completed.f{i}", f"Question {i}?", "data_contract", "checkout-completed")
        for i in range(5)
    ]

    async def fake_acompletion(*, model, api_key, messages, **kwargs):
        seen_prompts.append(messages[0]["content"])
        return _fake_response(json.dumps({"questions": not_shallow}))

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    bronze_documents = [{"source_component": "meeting.en.vtt", "content": "We discussed checkout."}]
    mentioned_data_contracts = [
        {
            "name": "checkout-completed",
            "producer": "checkout service",
            "consumer": "order history",
            "action": "forward-update",
        }
    ]
    await generate_data_contract_questions_for_batch("20260906", bronze_documents, mentioned_data_contracts)

    assert len(seen_prompts) == 1
    assert "checkout-completed" in seen_prompts[0]
    assert "checkout service" in seen_prompts[0]
    assert "order history" in seen_prompts[0]


def test_mentions_grounded_in_source_keeps_only_names_present_in_that_source_text():
    """`mentioned_components`/`mentioned_data_contracts` are drafted once over a whole
    ingestion_date batch's pooled transcript — this is how `write_document` learns which of
    the batch's mentions actually belong to one specific `source_component`'s row."""
    items = [
        {"name": "Checkout Service", "status": "new"},
        {"name": "Loyalty Service", "status": "new"},
    ]
    source_text = "Today we discussed the Checkout Service and its new payment flow."

    grounded = mentions_grounded_in_source(source_text, items)

    assert grounded == [{"name": "Checkout Service", "status": "new"}]


def test_mentions_grounded_in_source_is_case_insensitive_and_can_match_more_than_one():
    items = [
        {"name": "checkout service", "status": "unchanged"},
        {"name": "Loyalty Service", "status": "new"},
        {"name": "Notification Service", "status": "removed"},
    ]
    source_text = "CHECKOUT SERVICE now calls Loyalty Service before publishing the event."

    grounded = mentions_grounded_in_source(source_text, items)

    assert grounded == [
        {"name": "checkout service", "status": "unchanged"},
        {"name": "Loyalty Service", "status": "new"},
    ]


def test_mentions_grounded_in_source_returns_empty_list_when_nothing_matches():
    items = [{"name": "Checkout Service", "status": "new"}]
    assert mentions_grounded_in_source("A totally unrelated transcript about billing.", items) == []
