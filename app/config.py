from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


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

    # APP CONFIG
    app_name: str = "t-rag"
    environment: str = "dev"
    start_test_mode: bool = True # Open a UI section to monitor test metrics and token monitor
    log_level: str = "INFO"

    openai_api_key: str | None = None
    openai_model: str = "gpt-5.6-terra"
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-haiku-4-5"

    llm_fallback_order: list[Literal["openai", "anthropic"]] = ["openai", "anthropic"]

    max_architecture_pending_questions: int = 10
    max_data_contract_pending_questions: int = 10


settings = Settings()
