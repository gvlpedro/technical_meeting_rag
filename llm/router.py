import contextvars
import json
from datetime import UTC, datetime
from pathlib import Path
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

# Every real LLM call appends one JSON line here. This includes calls from the Silver/Gold
# graph, from /v1/completions, from the ACB test suites, and from this router's own
# retries. The "Test monitor" tab reads this file to show token and cost use. This is a
# plain append-only file, not a DB table. It needs to survive even when the call that
# triggered it fails partway through a Postgres transaction. Nothing downstream ever needs
# to query it with SQL-style joins.
LLM_USAGE_LOG = Path(settings.output_dir) / "llm_usage.jsonl"

# This is set once per request or session, at the caller's boundary (see `bind_tenant`). We
# read it here only to tag the usage log. We never use it to change what the call itself
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
    same model as the call it is judging."""
    errors: list[tuple[str, Exception]] = []

    for provider in providers or settings.llm_fallback_order:
        model = getattr(settings, _PROVIDER_MODEL_FIELD[provider])
        api_key = getattr(settings, _PROVIDER_API_KEY_FIELD[provider])
        try:
            response = await litellm.acompletion(model=model, api_key=api_key, messages=messages, **kwargs)
        except Exception as exc:  # noqa: BLE001 - if this provider fails, we fall through to the next one
            errors.append((provider, exc))
            continue
        _log_usage(provider, model, response)
        return response

    raise AllProvidersFailedError(errors)


def _log_usage(provider: str, model: str, response: Any) -> None:
    """Append one JSON line to `LLM_USAGE_LOG` for a successful call. This never raises an
    error. A broken cost lookup or a full disk must never break the actual LLM call. This
    function only tries to record that call, it must not stop it."""
    try:
        usage = getattr(response, "usage", None)
        if usage is None:
            # This is a faked response from a mocked-LLM test. Tests such as
            # tests/test_clarification_loop.py monkeypatch litellm.acompletion, not this
            # function, so their fake SimpleNamespace responses still reach here. There is
            # nothing real to log here. Logging a zero-token, zero-cost line would only add
            # noise from the fast, mocked test suite. It would dilute the Test Monitor tab's
            # real per-tenant totals.
            return
        cost_usd: float | None = None
        try:
            cost_usd = litellm.completion_cost(completion_response=response)
        except Exception:  # noqa: BLE001 - the pricing lookup can fail for an unlisted model
            pass

        entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "tenant": _tenant_ctx.get(),
            "provider": provider,
            "model": model,
            "input_tokens": getattr(usage, "prompt_tokens", None),
            "output_tokens": getattr(usage, "completion_tokens", None),
            "cost_usd": cost_usd,
            "cost_eur": cost_usd * USD_TO_EUR if cost_usd is not None else None,
        }
        LLM_USAGE_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(LLM_USAGE_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:  # noqa: BLE001 - logging usage must never break the call it is logging
        pass
