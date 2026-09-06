from types import SimpleNamespace
from unittest.mock import AsyncMock

import litellm
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _fake_response(content: str, model: str) -> SimpleNamespace:
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice], model=model)


def test_completions_returns_answer_from_first_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_response = _fake_response("hola desde openai", "gpt-4o-mini")
    monkeypatch.setattr(litellm, "acompletion", AsyncMock(return_value=fake_response))

    response = client.post(
        "/v1/completions",
        json={"messages": [{"role": "user", "content": "hola"}]},
    )

    assert response.status_code == 200
    assert response.json() == {"content": "hola desde openai", "model": "gpt-4o-mini"}


def test_completions_returns_502_when_all_providers_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_acompletion = AsyncMock(
        side_effect=[
            litellm.RateLimitError("openai down", llm_provider="openai", model="gpt-4o-mini"),
            litellm.RateLimitError(
                "anthropic down", llm_provider="anthropic", model="claude-3-5-sonnet-20241022"
            ),
        ]
    )
    monkeypatch.setattr(litellm, "acompletion", mock_acompletion)

    response = client.post(
        "/v1/completions",
        json={"messages": [{"role": "user", "content": "hola"}]},
    )

    assert response.status_code == 502
    assert "openai" in response.json()["detail"]
    assert "anthropic" in response.json()["detail"]


def test_completions_rejects_invalid_role() -> None:
    response = client.post(
        "/v1/completions",
        json={"messages": [{"role": "not-a-role", "content": "hola"}]},
    )

    assert response.status_code == 422
