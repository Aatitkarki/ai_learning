from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LAB_", env_file=".env", extra="ignore")
    data_dir: Path = Path("data")
    ollama_url: str = "http://127.0.0.1:11434"
    chat_model: str = ""
    embed_model: str = ""
    openai_base_url: str = ""
    openai_api_key: SecretStr = Field(default=SecretStr(""), exclude=True)
    openai_chat_model: str = ""
    openai_embed_model: str = ""
    openai_embed_revision: str = ""
    openai_token_parameter: Literal["max_completion_tokens", "max_tokens"] = "max_completion_tokens"
    openai_send_temperature: bool = False
    openai_stream_usage: bool = True
    qdrant_url: str = "http://127.0.0.1:6333"
    postgres_dsn: str = "postgresql://lab:local-lab-only@127.0.0.1:5434/lab"
    request_timeout: float = Field(default=120, ge=1, le=600)
    max_upload_bytes: int = 10 * 1024 * 1024
    max_text_chars: int = 2_000_000
    max_chunks: int = 20_000
    api_url: str = "http://127.0.0.1:8020"

    @field_validator("openai_base_url")
    @classmethod
    def compatible_api_url(cls, value: str) -> str:
        value = value.strip().rstrip("/")
        if not value:
            return ""
        parsed = urlparse(value)
        local = parsed.hostname in {"localhost", "127.0.0.1", "::1", "host.docker.internal"}
        if (
            not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or any(c.isspace() for c in value)
            or not (parsed.scheme == "https" or (parsed.scheme == "http" and local))
        ):
            raise ValueError(
                "Use an HTTPS API base URL, or HTTP for a local server, "
                "without embedded credentials or query parameters."
            )
        if parsed.path.endswith(("/chat/completions", "/embeddings", "/responses")):
            raise ValueError("Provide the API base URL (usually ending in /v1), not an endpoint.")
        return value

    @field_validator("ollama_url", "qdrant_url", "api_url")
    @classmethod
    def local_http_only(cls, value: str) -> str:
        parsed = urlparse(value)
        if (
            parsed.scheme != "http"
            or parsed.hostname
            not in {"localhost", "127.0.0.1", "::1", "host.docker.internal", "qdrant"}
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "Use a local HTTP service URL without credentials or query parameters."
            )
        return value.rstrip("/")


class LabError(Exception):
    def __init__(self, message: str, status: int = 400):
        self.message, self.status = message, status
        super().__init__(message)
