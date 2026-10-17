"""mcp_server/settings.py

All configuration for the MCP server comes from environment variables (or a local `.env`
file). Nothing here is a secret: the MCP server only needs to know WHERE the ML API lives.

The ML API address is deliberately a single, fixed setting. The LLM can never choose a URL,
because no tool parameter accepts one (see tools/schemas.py).
"""
from __future__ import annotations

from urllib.parse import urlparse

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- The ONE destination this server is allowed to talk to --------------------------
    ml_api_base_url: str = "http://localhost:8000"
    ml_api_timeout_seconds: float = 10.0

    # --- How this MCP server is exposed -------------------------------------------------
    mcp_transport: str = "streamable-http"  # or "stdio" (handy with the MCP Inspector)
    mcp_host: str = "127.0.0.1"             # inside Docker we set 0.0.0.0
    mcp_port: int = 8001
    # Host headers we accept. Protects against "DNS rebinding" (a malicious web page
    # tricking a browser into talking to a server on your laptop).
    mcp_allowed_hosts: str = "localhost:*,127.0.0.1:*,mcp-server:*"

    log_level: str = "INFO"

    @field_validator("ml_api_base_url")
    @classmethod
    def _check_url(cls, v: str) -> str:
        parsed = urlparse(v)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError("ML_API_BASE_URL must look like http://host:port")
        return v.rstrip("/")

    @field_validator("mcp_transport")
    @classmethod
    def _check_transport(cls, v: str) -> str:
        if v not in ("streamable-http", "stdio"):
            raise ValueError("MCP_TRANSPORT must be 'streamable-http' or 'stdio'")
        return v

    @property
    def allowed_hosts(self) -> list[str]:
        return [h.strip() for h in self.mcp_allowed_hosts.split(",") if h.strip()]


def get_settings() -> Settings:
    return Settings()
