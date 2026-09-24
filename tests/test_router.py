from types import SimpleNamespace
from unittest.mock import AsyncMock

import litellm
import pytest
from sqlalchemy import delete, select

from app.config import settings
from db.models import LlmCost
from db.session import async_session_factory
from llm.router import AllProvidersFailedError, bind_tenant, complete

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(autouse=True)
async def _clean_llm_costs():
    yield
    async with async_session_factory() as session:
        await session.execute(delete(LlmCost).where(LlmCost.tenant == "router-test-tenant"))
        await session.commit()


@pytest.fixture(autouse=True)
def fallback_order(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "llm_fallback_order", ["openai", "anthropic"])
    monkeypatch.setattr(settings, "openai_model", "gpt-4o-mini")
    monkeypatch.setattr(settings, "anthropic_model", "claude-3-5-sonnet-20241022")


@pytest.mark.parametrize(
    "first_error",
    [
        litellm.RateLimitError("rate limited", llm_provider="openai", model="gpt-4o-mini"),
        litellm.Timeout("timed out", model="gpt-4o-mini", llm_provider="openai"),
        litellm.AuthenticationError("bad key", llm_provider="openai", model="gpt-4o-mini"),
    ],
)
async def test_falls_back_to_second_provider_on_first_failure(
    monkeypatch: pytest.MonkeyPatch, first_error: Exception
) -> None:
    fake_response = object()
    mock_acompletion = AsyncMock(side_effect=[first_error, fake_response])
    monkeypatch.setattr(litellm, "acompletion", mock_acompletion)

    messages = [{"role": "user", "content": "hola"}]
    result = await complete(messages)

    assert result is fake_response
    assert mock_acompletion.call_count == 2

    first_call, second_call = mock_acompletion.call_args_list
    assert first_call.kwargs["model"] == "gpt-4o-mini"
    assert first_call.kwargs["messages"] == messages
    assert second_call.kwargs["model"] == "claude-3-5-sonnet-20241022"
    assert second_call.kwargs["messages"] == messages


async def test_all_providers_failing_raises_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_acompletion = AsyncMock(
        side_effect=[
            litellm.RateLimitError("openai down", llm_provider="openai", model="gpt-4o-mini"),
            litellm.RateLimitError(
                "anthropic down", llm_provider="anthropic", model="claude-3-5-sonnet-20241022"
            ),
        ]
    )
    monkeypatch.setattr(litellm, "acompletion", mock_acompletion)

    with pytest.raises(AllProvidersFailedError) as exc_info:
        await complete([{"role": "user", "content": "hola"}])

    assert "openai" in str(exc_info.value)
    assert "anthropic" in str(exc_info.value)
    assert mock_acompletion.call_count == 2


async def _caller_of_complete(messages: list[dict[str, str]]):
    """A uniquely-named wrapper so the test below can assert `complete()` recorded THIS exact
    function's name as `llm_costs.method`, without colliding with any other test's own
    call-site name in the same file."""
    return await complete(messages)


async def test_successful_call_writes_a_llm_costs_row_tagged_with_the_caller_and_tenant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_response = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20))
    monkeypatch.setattr(litellm, "acompletion", AsyncMock(return_value=fake_response))
    monkeypatch.setattr(litellm, "cost_per_token", lambda **kwargs: (0.001, 0.002))

    bind_tenant("router-test-tenant")
    messages = [{"role": "user", "content": "What is Order Service?"}]
    await _caller_of_complete(messages)

    async with async_session_factory() as session:
        rows = (
            await session.execute(select(LlmCost).where(LlmCost.tenant == "router-test-tenant"))
        ).scalars().all()

    assert len(rows) == 1
    row = rows[0]
    assert row.method == "_caller_of_complete"
    assert row.model == settings.openai_model
    assert "What is Order Service?" in row.prompt
    assert row.input_cost == pytest.approx(0.001 * 0.92)
    assert row.output_cost == pytest.approx(0.002 * 0.92)


async def test_mocked_response_with_no_usage_writes_no_llm_costs_row(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "acompletion", AsyncMock(return_value=object()))

    bind_tenant("router-test-tenant")
    await _caller_of_complete([{"role": "user", "content": "hola"}])

    async with async_session_factory() as session:
        rows = (
            await session.execute(select(LlmCost).where(LlmCost.tenant == "router-test-tenant"))
        ).scalars().all()
    assert rows == []
