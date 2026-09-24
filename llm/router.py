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

# This is the same fixed rate the ACB test suites already use for their own cost estimate
# (agents/stages/architecture_questions/testing/test_golden_set.py). It is not a live FX lookup. It is only
# meant to show the rough size of what a run cost.
USD_TO_EUR = 0.92

# This is set once per request or session, at the caller's boundary (see `bind_tenant`). We
# read it here only to tag `llm_costs.tenant`. We never use it to change what the call itself
# does. It defaults to "default", so a call made outside any frontend request (a script, a
# test) still logs a valid, if generic, tenant instead of `None`.
_tenant_ctx: contextvars.ContextVar[str] = contextvars.ContextVar("llm_router_tenant", default="default")


def bind_tenant(tenant: str) -> None:
    """Set the tenant that every later `complete()` call in this task or request logs under.

    This is a contextvar, not a parameter on `complete()` itself. `complete()` is called
    from deep inside `agents/graph.py`, `agents/service.py`, and
    `agents/stages/gold/service.py`. None of these otherwise need to know about tenants at
    all. Passing an explicit parameter through every one of those call sites, just to tag a
    log line, would be a much larger and riskier change than this one call at each real
    entry point (an HTTP request, a graph run)."""
    _tenant_ctx.set(tenant)


class AllProvidersFailedError(RuntimeError):
    def __init__(self, errors: list[tuple[str, Exception]]) -> None:
        detail = "; ".join(f"{provider}: {error!r}" for provider, error in errors)
        super().__init__(f"All LLM providers failed: {detail}")
        self.errors = errors


async def complete(
    messages: list[dict[str, str]], providers: list[str] | None = None, **kwargs: Any
) -> Any:
    """Try each provider in `providers`, in order, until one succeeds. If `providers` is not
    given, use `settings.llm_fallback_order`. Pass an explicit `providers` order to force a
    specific provider first. For example, use this for an evaluator that must not run on the
    same model as the call it is judging.

    `method` (the `llm_costs.method` column) is read off the call stack right here, before
    anything else runs — `sys._getframe(1).f_code.co_name` is the function that directly
    called `complete()`. This is captured instead of asking every call site to pass a
    `method="..."` kwarg, for the same reason `bind_tenant`'s own docstring already gives for
    tenant: `complete()` is called from over a dozen places across `agents/graph.py` and every
    stage's `service.py`, none of which need to know they are being logged. Introspection also
    can never drift out of sync with a function that gets renamed later, the way a hand-typed
    label could."""
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
    """Writes one row to `llm_costs` for a successful call. This never raises an error. A
    broken cost lookup, a DB hiccup, or an unlisted model's missing price must never break the
    actual LLM call this is only trying to record."""
    try:
        usage = getattr(response, "usage", None)
        if usage is None:
            # This is a faked response from a mocked-LLM test. Tests such as
            # tests/test_clarification_loop.py monkeypatch litellm.acompletion, not this
            # function, so their fake SimpleNamespace responses still reach here. There is
            # nothing real to log here. Writing a zero-token, zero-cost row would only add
            # noise from the fast, mocked test suite, diluting the Monitor tab's real,
            # per-tenant call log.
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

        # Local import: `db.session`/`db.models` importing back into `llm.router` (both are
        # imported from `agents/graph.py` and elsewhere) would risk a circular import at
        # module load time. A local import only runs once this function is actually called.
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
