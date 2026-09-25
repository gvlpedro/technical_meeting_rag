from typing import Literal

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class FrontendUser(BaseModel):
    """One login the Streamlit frontend accepts. There is no user table, no signup, and no
    password hashing. This is a fixed, hardcoded list of logins. It is enough for a project
    of this size. See the login page in `frontend/` for how it is used.

    `tenant` is what actually keeps data separate between logins. Every row this user
    uploads or asks about is tagged with its tenant. Every query the frontend makes also
    filters by tenant."""

    username: str
    password: str
    tenant: str


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


    # If there is no token, logfire.configure(send_to_logfire="if-token-present") stays
    # local-only. It only prints to the console and makes no network calls. This is a safe
    # default for dev and CI.
    logfire_token: str | None = None

    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5433/technical_meeting_rag"
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384
    # A local cross-encoder for `agents.stages.gold.service.top_k_gold_evolution`'s optional
    # `rerank=True` step. Local, like `embedding_model` above — no external provider, no extra
    # API key. `ms-marco-MiniLM-L-6-v2` is the standard small, CPU-friendly cross-encoder for
    # passage reranking (see `.tmp/advanced_techniques.md` §3).
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    chunk_size_tokens: int = 400
    chunk_overlap_tokens: int = 50
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

    max_upload_file_bytes: int = 50 * 1024 * 1024  # 50 MB

    # These are hardcoded example logins for the Streamlit frontend.
    # See the "Frontend usage" section in GETTING_STARTED.md. Each login maps to its own tenant. The tenant
    # is what actually keeps data separate
    frontend_users: list[FrontendUser] = [
        FrontendUser(username="pepe", password="1234", tenant="lidr"),
        FrontendUser(username="peter", password="123", tenant="lotus"),
        FrontendUser(username="martin", password="123", tenant="lotus"),
    ]


settings = Settings()
