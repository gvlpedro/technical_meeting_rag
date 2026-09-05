from unittest.mock import AsyncMock

import litellm
import pytest

from app.config import settings
from llm.router import AllProvidersFailedError, complete

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


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
