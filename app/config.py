from typing import Literal

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class FrontendUser(BaseModel):
    """One login for the Streamlit frontend. The list is fixed and hardcoded. There is no
    user table, no signup, and no password hash.

    `tenant` keeps data separate between logins. Every row tags its tenant. Every query
    filters by tenant."""

    username: str
    password: str
    tenant: str


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


    # With no token, logfire stays local-only: console output, no network calls. Safe
    # default for dev and CI.
    logfire_token: str | None = None

    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5433/technical_meeting_rag"
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384
    # Local cross-encoder for the optional `rerank=True` step in
    # `agents.stages.gold.service.top_k_gold_evolution`. No external API.
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    chunk_size_tokens: int = 400
    chunk_overlap_tokens: int = 50
    output_dir: str = "output"

    # APP CONFIG
    app_name: str = "t-rag"
    environment: str = "dev"
    start_test_mode: bool = True # Shows a UI tab with test metrics and token usage
    log_level: str = "INFO"

    openai_api_key: str | None = None
    openai_model: str = "gpt-5.6-terra"
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-haiku-4-5"

    llm_fallback_order: list[Literal["openai", "anthropic"]] = ["openai", "anthropic"]

    max_architecture_pending_questions: int = 10
    max_data_contract_pending_questions: int = 10

    max_upload_file_bytes: int = 50 * 1024 * 1024  # 50 MB

    # Cost/abuse guardrail
    max_llm_calls_per_window: int = 20
    llm_rate_limit_window_seconds: int = 60

    # Hardcoded example logins. See GETTING_STARTED.md, "Frontend usage". Each login maps
    # to its own tenant, which keeps data separate.
    frontend_users: list[FrontendUser] = [
        FrontendUser(username="peter", password="123", tenant="lotus"),
        FrontendUser(username="martin", password="123", tenant="lotus"),
        FrontendUser(username="mike", password="444", tenant="ibm"),
        FrontendUser(username="fer", password="1234", tenant="trial"),
    ]


settings = Settings()
