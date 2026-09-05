from typing import Any

import litellm

from app.config import settings

_PROVIDER_MODEL_FIELD = {
    "openai": "openai_model",
    "anthropic": "anthropic_model",
}
_PROVIDER_API_KEY_FIELD = {
    "openai": "openai_api_key",
    "anthropic": "anthropic_api_key",
}


class AllProvidersFailedError(RuntimeError):
    def __init__(self, errors: list[tuple[str, Exception]]) -> None:
        detail = "; ".join(f"{provider}: {error!r}" for provider, error in errors)
        super().__init__(f"All LLM providers failed: {detail}")
        self.errors = errors


async def complete(messages: list[dict[str, str]], **kwargs: Any) -> Any:
    """Try each provider in `settings.llm_fallback_order`, in order, until one succeeds."""
    errors: list[tuple[str, Exception]] = []

    for provider in settings.llm_fallback_order:
        model = getattr(settings, _PROVIDER_MODEL_FIELD[provider])
        api_key = getattr(settings, _PROVIDER_API_KEY_FIELD[provider])
        try:
            return await litellm.acompletion(model=model, api_key=api_key, messages=messages, **kwargs)
        except Exception as exc:  # noqa: BLE001 - any provider failure falls through to the next one
            errors.append((provider, exc))

    raise AllProvidersFailedError(errors)
