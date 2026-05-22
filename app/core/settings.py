"""Centralised application settings.

Settings are loaded from environment variables (and a `.env` file in dev) using
``pydantic-settings``. All other modules MUST import the settings instance from
here so we have a single source of truth.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

LLMProvider = Literal["mock", "anthropic", "openai", "google", "azure_openai", "bedrock"]


class Settings(BaseSettings):
    """Application settings (env-driven, `.env`-aware)."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---- App ---------------------------------------------------------------
    app_env: Literal["dev", "staging", "prod"] = "dev"
    app_log_level: str = "INFO"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_secret_key: str = "change-me-please"

    # ---- LLM provider ------------------------------------------------------
    llm_provider: LLMProvider = "mock"
    llm_model: str = "anthropic:claude-sonnet-4-5-20250929"

    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    google_api_key: str | None = None
    azure_openai_api_key: str | None = None
    azure_openai_endpoint: str | None = None
    openai_api_version: str | None = None

    # ---- Filesystem layout -------------------------------------------------
    workspace_root: Path = Field(default=Path("./workspace"))
    memory_root: Path = Field(default=Path("./memories"))
    skills_root: Path = Field(default=Path("./skills"))

    # ---- Storage -----------------------------------------------------------
    database_url: str = "sqlite:///./cosmetic_deepstat.db"
    s3_endpoint_url: str | None = None
    s3_bucket: str | None = None
    s3_access_key: str | None = None
    s3_secret_key: str | None = None
    s3_region: str | None = None
    vector_store_backend: Literal["none", "chroma", "qdrant", "pgvector"] = "none"

    # ---- Security ----------------------------------------------------------
    allow_data_mutation_in_raw: bool = False
    pseudonymize_subjects: bool = True

    # ---- Observability -----------------------------------------------------
    enable_audit_trail: bool = True
    audit_log_path: Path = Field(default=Path("./workspace/_audit/audit.jsonl"))

    langsmith_tracing: bool = False
    langsmith_api_key: str | None = None
    langsmith_project: str = "cosmetic-deepstat-agents"

    # ---- Helpers -----------------------------------------------------------
    @property
    def workspace_root_abs(self) -> Path:
        return self.workspace_root.resolve()

    @property
    def memory_root_abs(self) -> Path:
        return self.memory_root.resolve()

    @property
    def skills_root_abs(self) -> Path:
        return self.skills_root.resolve()

    def ensure_dirs(self) -> None:
        """Create the workspace / memory / audit directories if missing."""
        self.workspace_root_abs.mkdir(parents=True, exist_ok=True)
        self.memory_root_abs.mkdir(parents=True, exist_ok=True)
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton ``Settings`` instance."""
    s = Settings()
    s.ensure_dirs()
    return s
