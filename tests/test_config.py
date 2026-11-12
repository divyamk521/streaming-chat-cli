"""Tests for application configuration."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from streaming_chat.config import Settings


def test_default_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify application defaults."""
    environment_variables = [
        "LLM_PROVIDER",
        "GROQ_API_KEY",
        "GROQ_MODEL",
        "OLLAMA_BASE_URL",
        "OLLAMA_MODEL",
        "LOG_LEVEL",
        "CACHE_DIR",
        "MAX_RETRIES",
        "RETRY_BASE_DELAY",
    ]

    for variable in environment_variables:
        monkeypatch.delenv(variable, raising=False)

    settings = Settings(_env_file=None)

    assert settings.llm_provider == "groq"
    assert settings.groq_model == "openai/gpt-oss-20b"
    assert settings.ollama_base_url == "http://localhost:11434"
    assert settings.log_level == "INFO"
    assert settings.cache_dir == Path(".cache/responses")
    assert settings.max_retries == 3
    assert settings.retry_base_delay == 1.0
    assert settings.groq_api_key is None


def test_load_settings_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify settings are loaded from environment variables."""
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434/")
    monkeypatch.setenv("MAX_RETRIES", "5")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")

    settings = Settings(_env_file=None)

    assert settings.llm_provider == "ollama"
    assert settings.ollama_base_url == "http://localhost:11434"
    assert settings.max_retries == 5
    assert settings.log_level == "DEBUG"


def test_groq_api_key_is_masked() -> None:
    """Verify API keys are represented as masked secrets."""
    settings = Settings(
        _env_file=None,
        groq_api_key="test-secret-key",
    )

    assert str(settings.groq_api_key) == "**********"
    assert settings.groq_api_key.get_secret_value() == "test-secret-key"


def test_require_groq_api_key_returns_valid_key() -> None:
    """Verify a configured Groq API key can be retrieved."""
    settings = Settings(
        _env_file=None,
        groq_api_key="test-secret-key",
    )

    assert settings.require_groq_api_key() == "test-secret-key"


def test_require_groq_api_key_rejects_missing_key() -> None:
    """Verify missing Groq credentials produce an actionable error."""
    settings = Settings(_env_file=None)

    with pytest.raises(ValueError, match="GROQ_API_KEY is required"):
        settings.require_groq_api_key()


def test_require_groq_api_key_rejects_placeholder() -> None:
    """Verify example credentials cannot be used as real API keys."""
    settings = Settings(
        _env_file=None,
        groq_api_key="your_groq_api_key_here",
    )

    with pytest.raises(ValueError, match="placeholder"):
        settings.require_groq_api_key()


def test_invalid_provider_is_rejected() -> None:
    """Verify unsupported providers are rejected."""
    with pytest.raises(ValidationError):
        Settings(_env_file=None, llm_provider="invalid")


def test_negative_retry_count_is_rejected() -> None:
    """Verify retry count cannot be negative."""
    with pytest.raises(ValidationError):
        Settings(_env_file=None, max_retries=-1)


def test_retry_count_above_limit_is_rejected() -> None:
    """Verify excessive retry counts are rejected."""
    with pytest.raises(ValidationError):
        Settings(_env_file=None, max_retries=11)


def test_zero_retry_delay_is_rejected() -> None:
    """Verify retry delay must be positive."""
    with pytest.raises(ValidationError):
        Settings(_env_file=None, retry_base_delay=0)


def test_invalid_ollama_url_is_rejected() -> None:
    """Verify invalid Ollama URLs are rejected."""
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            ollama_base_url="ftp://localhost:11434",
        )
