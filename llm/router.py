import contextvars
import sys
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

USD_TO_EUR = 0.92

# FastAPI asyncio.Task has there own tenant
_tenant_ctx: contextvars.ContextVar[str] = contextvars.ContextVar("llm_router_tenant", default="default")


def bind_tenant(tenant: str) -> None:
    """Set the tenant"""
    _tenant_ctx.set(tenant)


class AllProvidersFailedError(RuntimeError):
    def __init__(self, errors: list[tuple[str, Exception]]) -> None:
        detail = "; ".join(f"{provider}: {error!r}" for provider, error in errors)
        super().__init__(f"All LLM providers failed: {detail}")
        self.errors = errors


async def complete(
    messages: list[dict[str, str]], providers: list[str] | None = None, **kwargs: Any
) -> Any:
    """Try each provider in `providers`, in order, until one succeeds."""
    method = sys._getframe(1).f_code.co_name
    errors: list[tuple[str, Exception]] = []

    for provider in providers or settings.llm_fallback_order:
        model = getattr(settings, _PROVIDER_MODEL_FIELD[provider])
        api_key = getattr(settings, _PROVIDER_API_KEY_FIELD[provider])
        try:
            response = await litellm.acompletion(model=model, api_key=api_key, messages=messages, **kwargs)
        except Exception as exc:  # noqa: BLE001 - if this provider fails, we fall through to the next one
            errors.append((provider, exc))
            continue
        await _log_usage(method, model, messages, response)
        return response

    raise AllProvidersFailedError(errors)


async def _log_usage(method: str, model: str, messages: list[dict[str, str]], response: Any) -> None:
    """Writes one row to `llm_costs` for a successful call."""
    try:
        usage = getattr(response, "usage", None)
        if usage is None:
            return

        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        input_cost_usd = output_cost_usd = 0.0
        try:
            input_cost_usd, output_cost_usd = litellm.cost_per_token(
                model=model, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
            )
        except Exception:  # noqa: BLE001 - the pricing lookup can fail for an unlisted model
            pass

        from db.models import LlmCost
        from db.session import async_session_factory

        prompt_text = "\n\n".join(f"[{m.get('role', '?')}] {m.get('content', '')}" for m in messages)
        async with async_session_factory() as session:
            session.add(
                LlmCost(
                    tenant=_tenant_ctx.get(),
                    method=method,
                    model=model,
                    prompt=prompt_text,
                    input_cost=input_cost_usd * USD_TO_EUR,
                    output_cost=output_cost_usd * USD_TO_EUR,
                )
            )
            await session.commit()
    except Exception:  # noqa: BLE001 - logging usage must never break the call it is logging
        pass
