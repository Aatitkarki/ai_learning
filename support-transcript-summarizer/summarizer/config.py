from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen2.5:3b"
    model_timeout_seconds: float = Field(default=180, gt=0, le=600)
    database_path: Path = Path("data/handovers.sqlite3")
    api_base_url: str = "http://127.0.0.1:8010"
