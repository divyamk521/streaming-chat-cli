"""Tests for persistent response caching."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from streaming_chat.cache import ResponseCache


@pytest.fixture
def messages() -> list[dict[str, str]]:
    """Provide a predictable conversation."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain Python."},
    ]


@pytest.fixture
def cache(tmp_path: Path) -> ResponseCache:
    """Provide a temporary response cache."""
    return ResponseCache(tmp_path / "cache")


def test_cache_miss_returns_none(
    cache: ResponseCache,
    messages: list[dict[str, str]],
) -> None:
    """A missing entry should return None."""
    result = cache.get(messages, "groq", "test-model")

    assert result is None


def test_cache_stores_and_retrieves_response(
    cache: ResponseCache,
    messages: list[dict[str, str]],
) -> None:
    """A stored response should be retrievable."""
    cache.set(
        messages,
        "groq",
        "test-model",
        "Python is a programming language.",
        prompt_tokens=100,
        completion_tokens=20,
    )

    entry = cache.get(messages, "groq", "test-model")

    assert entry is not None
    assert entry.response == "Python is a programming language."
    assert entry.prompt_tokens == 100
    assert entry.completion_tokens == 20


def test_cache_preserves_unicode(
    cache: ResponseCache,
    messages: list[dict[str, str]],
) -> None:
    """Unicode responses should be stored without data loss."""
    response = "नमस्ते! Hello! 🌟"

    cache.set(messages, "groq", "test-model", response)

    entry = cache.get(messages, "groq", "test-model")

    assert entry is not None
    assert entry.response == response


def test_different_providers_use_different_entries(
    cache: ResponseCache,
    messages: list[dict[str, str]],
) -> None:
    """Different providers must not share cached responses."""
    cache.set(messages, "groq", "test-model", "Groq response")

    result = cache.get(messages, "ollama", "test-model")

    assert result is None


def test_different_models_use_different_entries(
    cache: ResponseCache,
    messages: list[dict[str, str]],
) -> None:
    """Different models must not share cached responses."""
    cache.set(messages, "groq", "model-a", "Model A response")

    result = cache.get(messages, "groq", "model-b")

    assert result is None


def test_different_messages_use_different_entries(
    cache: ResponseCache,
    messages: list[dict[str, str]],
) -> None:
    """Different conversation content must produce different cache entries."""
    cache.set(messages, "groq", "test-model", "First response")

    changed_messages = [
        {"role": "user", "content": "Explain Java."},
    ]

    result = cache.get(changed_messages, "groq", "test-model")

    assert result is None


def test_expired_entry_is_removed(
    tmp_path: Path,
    messages: list[dict[str, str]],
) -> None:
    """Expired cache entries should be deleted."""
    cache = ResponseCache(tmp_path / "cache", ttl_seconds=3600)

    cache.set(messages, "groq", "test-model", "Old response")

    cache_path = cache._cache_path(messages, "groq", "test-model")
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    old_timestamp = datetime.now(UTC) - timedelta(hours=2)
    payload["created_at"] = old_timestamp.isoformat()
    cache_path.write_text(json.dumps(payload), encoding="utf-8")

    result = cache.get(messages, "groq", "test-model")

    assert result is None
    assert not cache_path.exists()


def test_corrupted_cache_entry_is_removed(
    cache: ResponseCache,
    messages: list[dict[str, str]],
) -> None:
    """Invalid JSON should be treated as a cache miss."""
    cache_path = cache._cache_path(messages, "groq", "test-model")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text("{invalid-json", encoding="utf-8")

    result = cache.get(messages, "groq", "test-model")

    assert result is None
    assert not cache_path.exists()


def test_cache_rejects_empty_provider(
    cache: ResponseCache,
    messages: list[dict[str, str]],
) -> None:
    """Provider names cannot be empty."""
    with pytest.raises(ValueError, match="provider cannot be empty"):
        cache.get(messages, " ", "test-model")


def test_cache_rejects_empty_model(
    cache: ResponseCache,
    messages: list[dict[str, str]],
) -> None:
    """Model names cannot be empty."""
    with pytest.raises(ValueError, match="model cannot be empty"):
        cache.get(messages, "groq", " ")


def test_cache_rejects_negative_token_count(
    cache: ResponseCache,
    messages: list[dict[str, str]],
) -> None:
    """Negative token counts cannot be cached."""
    with pytest.raises(ValueError, match="non-negative integer"):
        cache.set(
            messages,
            "groq",
            "test-model",
            "Response",
            prompt_tokens=-1,
        )


def test_cache_rejects_boolean_token_count(
    cache: ResponseCache,
    messages: list[dict[str, str]],
) -> None:
    """Boolean values must not be accepted as token counts."""
    with pytest.raises(ValueError, match="non-negative integer"):
        cache.set(
            messages,
            "groq",
            "test-model",
            "Response",
            completion_tokens=True,
        )


def test_cache_rejects_negative_ttl(tmp_path: Path) -> None:
    """Cache lifetime cannot be negative."""
    with pytest.raises(ValueError, match="non-negative integer"):
        ResponseCache(tmp_path / "cache", ttl_seconds=-1)


def test_cache_creates_directory_automatically(
    tmp_path: Path,
    messages: list[dict[str, str]],
) -> None:
    """The cache directory should be created when storing an entry."""
    cache_dir = tmp_path / "new" / "nested" / "cache"
    cache = ResponseCache(cache_dir)

    cache.set(messages, "groq", "test-model", "Cached response")

    assert cache_dir.exists()
    assert cache.get(messages, "groq", "test-model") is not None
