"""Fast, mocked-LLM tests for the architecture-questions stage, in
`agents/stages/architecture_questions/`. Real-LLM quality and selectivity checks for this
same stage live in `agents/stages/architecture_questions/testing/`'s golden set instead. This file only tests
the mechanical plumbing: audit-file writing, prompt content, and batching. It uses a faked
`litellm.acompletion`. So it belongs in the fast `make test` suite."""

import json
from types import SimpleNamespace

import litellm
import pytest

from agents.stages.architecture_questions.prompts import build_architecture_question_generation_prompt
from agents.stages.architecture_questions.service import (
    generate_architecture_questions_for_batch,
    previous_architecture_context,
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


_SAMPLE_ADR = """# ADR — Introduce Payment Gateway

## 1. ADR

### Context

Payment Gateway is a new component.

## 2. Previous Architecture

The previous architecture is not described by the transcript and clarifications.

## 3. Target Architecture

```mermaid
flowchart LR
    CheckoutService[Checkout Service] --> PaymentGateway[Payment Gateway]
```

## 4. Affected Components

| Component | Change | Description |
|---|---|---|
| Payment Gateway | **NEW** | Processes card payments. |

## 5. Affected Data Contracts

No data contract changes were confirmed by the transcript and clarifications for this change.
"""


def test_previous_architecture_context_includes_diagram_and_affected_components_only():
    context = previous_architecture_context(_SAMPLE_ADR)
    assert context.startswith("## 3. Target Architecture")
    assert "flowchart LR" in context
    assert "## 4. Affected Components" in context
    assert "Payment Gateway" in context
    # This stops before section 5. Data contracts are not part of "known architecture" context.
    assert "## 5. Affected Data Contracts" not in context


def test_previous_architecture_context_is_empty_when_there_is_no_previous_adr():
    assert previous_architecture_context(None) == ""


# --- Prompt content (agents/stages/architecture_questions/prompts.py) ------------------------


def test_architecture_question_prompt_with_no_architecture_says_none_known():
    messages = build_architecture_question_generation_prompt("TEMPLATE", "TRANSCRIPT", architecture_diagram="")
    content = messages[0]["content"]

    assert "TEMPLATE" in content
    assert "TRANSCRIPT" in content
    assert "No existing architecture was provided." in content
    assert "flowchart" not in content


def test_architecture_question_prompt_with_architecture_includes_the_diagram_verbatim():
    diagram = "flowchart LR\n  thunder[Thunder] --> phoenix[Phoenix]"
    messages = build_architecture_question_generation_prompt(
        "TEMPLATE", "TRANSCRIPT", architecture_diagram=diagram
    )
    content = messages[0]["content"]

    assert diagram in content
    assert "No existing architecture was provided." not in content


def test_architecture_question_prompt_identifies_but_does_not_specify_data_contracts():
    content = build_architecture_question_generation_prompt("TEMPLATE", "TRANSCRIPT", architecture_diagram="")[
        0
    ]["content"]

    # Stage 1 identifies contracts: name, producer, consumer, action.
    assert "mentioned_data_contracts" in content
    # But it defers full ODCS depth to the next stage.
    assert "next stage" in content.lower()
