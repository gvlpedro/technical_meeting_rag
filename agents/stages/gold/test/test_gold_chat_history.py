"""Fast, mocked-LLM tests for conversational memory in the Gold chat (`.tmp/advanced_techniques.md`
§8). `test_gold.py` deliberately keeps zero LLM involvement, so this file — which needs to fake
`litellm.acompletion` the same way `tests/test_clarification_loop.py` does — lives separately.

`_format_history_block` is pure and gets its own no-LLM tests. `answer_question` and
`answer_evolution_question` each get one test proving `history` reaches the actual prompt sent
to the LLM, and one proving no history block is added when `history` is empty/`None` — so a
caller that never passes `history` (every call site before this feature) keeps behaving exactly
as before.
"""

import json
from datetime import date
from types import SimpleNamespace

import litellm
import pytest

from agents.stages.gold.service import (
    MAX_HISTORY_MESSAGES,
    _format_history_block,
    answer_evolution_question,
    answer_question,
)
from db.models import GoldEvolution
from db.session import async_session_factory

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _fake_response(content: str) -> SimpleNamespace:
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice], model="fake")


def _fake_answer(answer: str) -> SimpleNamespace:
    """`answer_question`/`answer_evolution_question` now expect a `GroundedAnswer`-shaped
    JSON response (`{"answer": ..., "citations": [...]}`), not bare text — see
    `.tmp/tasks2.md` task 1. These history tests only care about the prompt SENT to the LLM,
    not the citations, so an empty citations list is enough."""
    return _fake_response(json.dumps({"answer": answer, "citations": []}))


def _row(canonical_name: str = "Order Service", version: int = 1, operation: str = "new") -> GoldEvolution:
    return GoldEvolution(
        entity_type="component",
        entity_id="order-service",
        canonical_name=canonical_name,
        version=version,
        operation=operation,
        narrative="Order Service was introduced to own order placement.",
        source_component="order_fulfillment_platform.en.vtt",
        ingestion_date=date(2026, 1, 15),
        authored_by="alice",
        tenant="default",
    )


def test_format_history_block_is_empty_for_no_history():
    assert _format_history_block(None) == ""
    assert _format_history_block([]) == ""


def test_format_history_block_renders_role_and_content_oldest_last_message_kept():
    block = _format_history_block([("user", "What is Order Service?"), ("assistant", "It owns order placement.")])
    assert "User: What is Order Service?" in block
    assert "Assistant: It owns order placement." in block


def test_format_history_block_keeps_only_the_last_max_history_messages():
    history = [("user", f"turn {i}") for i in range(MAX_HISTORY_MESSAGES + 4)]
    block = _format_history_block(history)
    assert f"turn {MAX_HISTORY_MESSAGES + 3}" in block
    assert "turn 0" not in block
    assert block.count("User:") == MAX_HISTORY_MESSAGES


async def test_answer_question_includes_history_in_the_prompt_sent_to_the_llm(monkeypatch):
    captured = {}

    async def fake_acompletion(*, model, api_key, messages, **kwargs):
        captured["content"] = messages[0]["content"]
        return _fake_answer("It was Order Service, introduced to own order placement.")

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    history = [("user", "What is Order Service?"), ("assistant", "It owns order placement.")]
    # A real session is required now: `answer_question` always checks for successors
    # (`get_successors`) for every component row, regardless of its own payload contents.
    async with async_session_factory() as session:
        await answer_question(session, "And who approved it?", [_row()], history=history)

    assert "Previous conversation" in captured["content"]
    assert "What is Order Service?" in captured["content"]
    assert "And who approved it?" in captured["content"]


async def test_answer_question_omits_history_block_when_no_history_given(monkeypatch):
    captured = {}

    async def fake_acompletion(*, model, api_key, messages, **kwargs):
        captured["content"] = messages[0]["content"]
        return _fake_answer("Order Service owns order placement.")

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    async with async_session_factory() as session:
        await answer_question(session, "What is Order Service?", [_row()])

    assert "Previous conversation" not in captured["content"]


async def test_answer_evolution_question_includes_history_in_the_prompt_sent_to_the_llm(monkeypatch):
    captured = {}

    async def fake_acompletion(*, model, api_key, messages, **kwargs):
        captured["content"] = messages[0]["content"]
        return _fake_answer("Order Service was introduced, then later updated for idempotency.")

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    history = [("user", "Tell me about Order Service."), ("assistant", "It was introduced in step one.")]
    await answer_evolution_question("And how did it change since then?", "Order Service", [_row()], history=history)

    assert "Previous conversation" in captured["content"]
    assert "Tell me about Order Service." in captured["content"]


async def test_answer_evolution_question_omits_history_block_when_no_history_given(monkeypatch):
    captured = {}

    async def fake_acompletion(*, model, api_key, messages, **kwargs):
        captured["content"] = messages[0]["content"]
        return _fake_answer("Order Service was introduced to own order placement.")

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    await answer_evolution_question("How did Order Service evolve?", "Order Service", [_row()])

    assert "Previous conversation" not in captured["content"]
