from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "t-rag"
    environment: str = "dev"
    start_test_mode: bool = True # Open a UI section to monitor test metrics and token monitor
    log_level: str = "INFO"

    openai_api_key: str | None = None
    # gpt-5.6-terra: same 10/10 structural pass rate as gpt-5.6-sol on testing_questions_acb's
    # golden set, with a higher average Critic score (less padding) at half the cost — see
    # doc/cost_analysis.md. Needs reasoning_effort="none" alongside temperature=0 (see
    # agents/service.py).
    openai_model: str = "gpt-5.6-terra"
    anthropic_api_key: str | None = None
    # claude-haiku-4-5: not chosen for quality (36.9 avg Critic score, well below terra/sol) —
    # it's the only Anthropic model verified to actually accept temperature=0 for these calls.
    # claude-opus-5 (the previous default) rejects temperature=0 outright with no escape hatch,
    # so it silently broke this fallback; a mediocre answer here beats a hard failure. See
    # doc/cost_analysis.md.
    anthropic_model: str = "claude-haiku-4-5"
    llm_fallback_order: list[Literal["openai", "anthropic"]] = ["openai", "anthropic"]

    # No token -> logfire.configure(send_to_logfire="if-token-present") stays local-only
    # (console output, no network calls) — safe default for dev and CI.
    logfire_token: str | None = None

    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5433/technical_meeting_rag"
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384
    chunk_size_tokens: int = 400
    chunk_overlap_tokens: int = 50
    input_dir: str = "input"
    output_dir: str = "output"


settings = Settings()
