"""Persistent response caching."""

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from streaming_chat.providers.base import ChatMessage

CACHE_VERSION = 1
DEFAULT_CACHE_TTL_SECONDS = 86_400


@dataclass(frozen=True, slots=True)
class CacheEntry:
    """A cached model response and its token usage."""

    response: str
    prompt_tokens: int | None
    completion_tokens: int | None
    created_at: datetime


class ResponseCache:
    """Store and retrieve model responses from a local directory."""

    def __init__(
        self,
        cache_dir: Path,
        ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS,
    ) -> None:
        """Initialize the response cache."""
        if isinstance(ttl_seconds, bool) or not isinstance(ttl_seconds, int):
            raise ValueError("ttl_seconds must be a non-negative integer.")

        if ttl_seconds < 0:
            raise ValueError("ttl_seconds must be a non-negative integer.")

        self._cache_dir = Path(cache_dir)
        self._ttl_seconds = ttl_seconds

    def get(
        self,
        messages: list[ChatMessage],
        provider: str,
        model: str,
    ) -> CacheEntry | None:
        """Retrieve a valid cached response, if one exists."""
        cache_path = self._cache_path(messages, provider, model)

        if not cache_path.exists():
            return None

        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            entry = self._parse_entry(payload)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            self._delete_file(cache_path)
            return None

        age_seconds = (datetime.now(UTC) - entry.created_at).total_seconds()

        if age_seconds >= self._ttl_seconds:
            self._delete_file(cache_path)
            return None

        return entry

    def set(
        self,
        messages: list[ChatMessage],
        provider: str,
        model: str,
        response: str,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
    ) -> None:
        """Persist a completed model response."""
        self._validate_token_count(prompt_tokens, "prompt_tokens")
        self._validate_token_count(completion_tokens, "completion_tokens")

        if not isinstance(response, str):
            raise ValueError("response must be a string.")

        cache_path = self._cache_path(messages, provider, model)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        payload = {
            "version": CACHE_VERSION,
            "response": response,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "created_at": datetime.now(UTC).isoformat(),
        }

        temporary_path: Path | None = None

        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self._cache_dir,
                prefix=f".{cache_path.stem}.",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                json.dump(payload, temporary_file, ensure_ascii=False)

            os.replace(temporary_path, cache_path)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()

    def _cache_path(
        self,
        messages: list[ChatMessage],
        provider: str,
        model: str,
    ) -> Path:
        """Build a deterministic cache filename."""
        if not provider.strip():
            raise ValueError("provider cannot be empty.")

        if not model.strip():
            raise ValueError("model cannot be empty.")

        cache_data = {
            "version": CACHE_VERSION,
            "provider": provider,
            "model": model,
            "messages": messages,
        }

        serialized = json.dumps(
            cache_data,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )

        cache_key = hashlib.sha256(serialized.encode("utf-8")).hexdigest()

        return self._cache_dir / f"{cache_key}.json"

    @staticmethod
    def _parse_entry(payload: object) -> CacheEntry:
        """Validate and convert a stored JSON entry."""
        if not isinstance(payload, dict):
            raise ValueError("Invalid cache entry.")

        if payload.get("version") != CACHE_VERSION:
            raise ValueError("Unsupported cache version.")

        response = payload.get("response")
        prompt_tokens = payload.get("prompt_tokens")
        completion_tokens = payload.get("completion_tokens")
        created_at_value = payload.get("created_at")

        if not isinstance(response, str):
            raise ValueError("Invalid cached response.")

        ResponseCache._validate_token_count(prompt_tokens, "prompt_tokens")
        ResponseCache._validate_token_count(completion_tokens, "completion_tokens")

        if not isinstance(created_at_value, str):
            raise ValueError("Invalid cache timestamp.")

        created_at = datetime.fromisoformat(created_at_value)

        if created_at.tzinfo is None:
            raise ValueError("Cache timestamp must include a timezone.")

        return CacheEntry(
            response=response,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            created_at=created_at,
        )

    @staticmethod
    def _validate_token_count(
        value: int | None,
        name: str,
    ) -> None:
        """Validate an optional token count."""
        if value is None:
            return

        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer or None.")

    @staticmethod
    def _delete_file(cache_path: Path) -> None:
        """Remove an invalid or expired cache file."""
        try:
            cache_path.unlink(missing_ok=True)
        except OSError:
            pass
