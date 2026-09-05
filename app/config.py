from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "t-rag"
    environment: str = "dev"
    start_test_mode: bool = True # Open a UI section to monitor test metrics and token monitor
    log_level: str = "INFO"

    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-3-5-sonnet-20241022"
    llm_fallback_order: list[Literal["openai", "anthropic"]] = ["openai", "anthropic"]


settings = Settings()
