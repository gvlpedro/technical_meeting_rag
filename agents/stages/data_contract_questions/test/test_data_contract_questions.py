"""Fast, mocked-LLM tests for the data-contract-questions stage, in
`agents/stages/data_contract_questions/`. Real-LLM quality and completeness checks for this
same stage live in `agents/stages/data_contract_questions/testing/`'s golden set instead. This file only
tests the mechanical plumbing: audit-file writing, prompt content, and the empty-contracts
short-circuit. It uses a faked `litellm.acompletion`."""

import json
from types import SimpleNamespace

import litellm
import pytest

from agents.stages.data_contract_questions.prompts import build_data_contract_question_generation_prompt
from agents.stages.data_contract_questions.service import generate_data_contract_questions_for_batch
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
    assert calls == 0  # No contracts were identified. So there is nothing to ask an LLM about.


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
    # 5 questions for 1 contract clears MIN_QUESTIONS_PER_CONTRACT. A shallower result would
    # trigger the shallow-retry loop. This test only cares about the first prompt sent.
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


# --- Prompt content (agents/stages/data_contract_questions/prompts.py) -----------------------


def test_data_contract_question_prompt_covers_odcs_depth_and_restrains_padding():
    content = build_data_contract_question_generation_prompt("REQUIREMENTS", "TRANSCRIPT", [])[0]["content"]

    # These are the ODCS-shaped categories a data-contract question should cover. Versioning
    # covers previous version, new version, breaking changes, and affected consumers. Schema
    # depth covers required and optional fields, and constraints. Quality covers freshness.
    for expected in (
        "Previous contract version",
        "New contract version",
        "break-change",
        "forward-update",
        "Affected consumers",
        "Required/optional",
        "Constraints",
        "Freshness",
        "duplicate",
    ):
        assert expected in content

    # This prompt has no denylist of governance or bookkeeping fields. Instead, it uses a
    # general restraint: don't ask about an ODCS property just because ODCS has it.
    assert "Do not ask for generic ODCS properties merely because they exist" in content


def test_data_contract_question_prompt_includes_identified_contracts():
    messages = build_data_contract_question_generation_prompt(
        "REQUIREMENTS",
        "TRANSCRIPT",
        [
            {
                "name": "processed-event",
                "producer": "Event Processor",
                "consumer": "Analytics",
                "action": "forward-update",
            }
        ],
    )
    content = messages[0]["content"]

    assert "processed-event" in content
    assert "Event Processor" in content
    assert "Analytics" in content
