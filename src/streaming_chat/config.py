"""Centralized application configuration."""

from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        validate_assignment=True,
    )

    llm_provider: Literal["groq", "ollama"] = "groq"

    groq_api_key: SecretStr | None = None
    groq_model: str = "openai/gpt-oss-20b"

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:3b"

    log_level: Literal[
        "DEBUG",
        "INFO",
        "WARNING",
        "ERROR",
        "CRITICAL",
    ] = "INFO"

    cache_dir: Path = Path(".cache/responses")

    max_retries: int = Field(default=3, ge=0, le=10)
    retry_base_delay: float = Field(default=1.0, gt=0, le=60)

    @field_validator("ollama_base_url")
    @classmethod
    def validate_ollama_base_url(cls, value: str) -> str:
        """Ensure the Ollama endpoint has a valid HTTP or HTTPS URL."""
        value = value.strip()
        parsed_url = urlsplit(value)

        if parsed_url.scheme not in {"http", "https"}:
            raise ValueError("Ollama URL must use HTTP or HTTPS.")

        if not parsed_url.netloc or parsed_url.hostname is None:
            raise ValueError("Ollama URL must contain a valid hostname.")

        try:
            parsed_url.port
        except ValueError as exc:
            raise ValueError("Ollama URL contains an invalid port.") from exc

        return value.rstrip("/")

    def require_groq_api_key(self) -> str:
        """Return the Groq API key or raise an actionable error."""
        if self.groq_api_key is None:
            raise ValueError(
                "GROQ_API_KEY is required when using the Groq provider. "
                "Add your key to the local .env file."
            )

        api_key = self.groq_api_key.get_secret_value().strip()

        if not api_key or api_key == "your_groq_api_key_here":
            raise ValueError("GROQ_API_KEY is missing or still contains the example placeholder.")

        return api_key


def get_settings() -> Settings:
    """Create and return validated application settings."""
    return Settings()
