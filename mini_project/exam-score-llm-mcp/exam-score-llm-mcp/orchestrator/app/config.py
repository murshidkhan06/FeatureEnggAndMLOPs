"""app/config.py

ALL configuration is read from environment variables (or an `.env` file next to where you
start the service). Switching from Groq to Ollama is one line: LLM_PROVIDER=ollama.

Secrets: GROQ_API_KEY is held as a `SecretStr`, so printing the settings object, logging
it, or returning it from an API shows '**********' instead of the real key.
"""
from __future__ import annotations

from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ---- LLM provider ------------------------------------------------------------------
    llm_provider: Literal["groq", "ollama"] = "groq"
    llm_timeout_seconds: float = 60.0
    llm_temperature: float = 0.0  # 0 = as deterministic as the provider allows

    groq_api_key: SecretStr | None = None
    groq_model: str = ""  # no hidden default: you must choose a model on purpose
    groq_base_url: str = "https://api.groq.com/openai/v1"

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = ""

    # ---- MCP server (the tool server) --------------------------------------------------
    mcp_server_url: str = "http://localhost:8001/mcp"
    mcp_timeout_seconds: float = 20.0

    # ---- HTTP surface ------------------------------------------------------------------
    cors_allow_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    max_request_bytes: int = 8192

    # ---- Safety limits -----------------------------------------------------------------
    max_tool_rounds: int = Field(default=3, ge=1, le=6)          # LLM <-> tool back-and-forths per question
    max_predictions_per_turn: int = Field(default=4, ge=1, le=10)  # e.g. comparing up to 4 students
    strict_provenance: bool = True  # reject feature values the user never actually wrote
    # Final scores in the ML project's data are validated to lie in 0-100. A regression model can still
    # extrapolate outside that range; we then show a warning (we never change the number).
    score_min: float = 0.0
    score_max: float = 100.0

    # ---- Conversation memory (in-memory; demo only) ------------------------------------
    conversation_ttl_seconds: int = 3600
    max_conversations: int = 200
    max_history_messages: int = 40

    log_level: str = "INFO"

    @field_validator("groq_base_url", "ollama_base_url", "mcp_server_url")
    @classmethod
    def _strip_slash(cls, v: str) -> str:
        if not v.startswith(("http://", "https://")):
            raise ValueError("must start with http:// or https://")
        return v.rstrip("/")

    @property
    def cors_origins(self) -> list[str]:
        return [o.strip() for o in self.cors_allow_origins.split(",") if o.strip()]

    @property
    def active_model(self) -> str:
        return self.groq_model if self.llm_provider == "groq" else self.ollama_model


def get_settings() -> Settings:
    return Settings()
