"""Tests for the application-level chat streaming service."""

import pytest

import streaming_chat.chat as chat_module
from streaming_chat.config import Settings
from streaming_chat.providers.base import (
    ChatMessage,
    LLMProvider,
    ProviderError,
    StreamChunk,
)


class FakeProvider(LLMProvider):
    """A predictable provider used for unit testing."""

    def __init__(
        self,
        chunks: list[StreamChunk] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.chunks = chunks or []
        self.error = error
        self.received_messages: list[ChatMessage] | None = None

    def stream(self, messages: list[ChatMessage]):
        """Return configured chunks or raise a configured error."""
        self.received_messages = messages

        if self.error is not None:
            raise self.error

        yield from self.chunks


def test_stream_chat_yields_chunks_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify the service forwards every response chunk in order."""
    expected_chunks = [
        StreamChunk(text="Hello"),
        StreamChunk(text=" there"),
        StreamChunk(text="!"),
    ]

    fake_provider = FakeProvider(chunks=expected_chunks)

    monkeypatch.setattr(
        chat_module,
        "get_llm",
        lambda settings: fake_provider,
    )

    messages: list[ChatMessage] = [{"role": "user", "content": "Say hello."}]

    actual_chunks = list(chat_module.stream_chat(messages))

    assert actual_chunks == expected_chunks


def test_stream_chat_passes_messages_to_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify conversation messages reach the selected provider."""
    fake_provider = FakeProvider()

    monkeypatch.setattr(
        chat_module,
        "get_llm",
        lambda settings: fake_provider,
    )

    messages: list[ChatMessage] = [
        {"role": "system", "content": "Be helpful."},
        {"role": "user", "content": "Explain Python."},
    ]

    list(chat_module.stream_chat(messages))

    assert fake_provider.received_messages is messages


def test_stream_chat_passes_settings_to_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify custom application settings reach the provider factory."""
    fake_provider = FakeProvider()
    settings = Settings(_env_file=None, llm_provider="ollama")
    captured_settings: list[Settings | None] = []

    def fake_get_llm(
        received_settings: Settings | None = None,
    ) -> LLMProvider:
        captured_settings.append(received_settings)
        return fake_provider

    monkeypatch.setattr(
        chat_module,
        "get_llm",
        fake_get_llm,
    )

    messages: list[ChatMessage] = [{"role": "user", "content": "Hello."}]

    list(chat_module.stream_chat(messages, settings))

    assert captured_settings == [settings]


def test_stream_chat_propagates_provider_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify provider errors reach the caller."""
    fake_provider = FakeProvider(error=ProviderError("Simulated provider failure."))

    monkeypatch.setattr(
        chat_module,
        "get_llm",
        lambda settings: fake_provider,
    )

    messages: list[ChatMessage] = [{"role": "user", "content": "Hello."}]

    with pytest.raises(ProviderError, match="Simulated provider failure"):
        list(chat_module.stream_chat(messages))


def test_stream_chat_rejects_empty_messages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify empty conversations are rejected before provider selection."""

    def unexpected_provider_call(
        settings: Settings | None = None,
    ) -> LLMProvider:
        raise AssertionError("Provider should not be initialized.")

    monkeypatch.setattr(
        chat_module,
        "get_llm",
        unexpected_provider_call,
    )

    with pytest.raises(ValueError, match="At least one chat message"):
        list(chat_module.stream_chat([]))
