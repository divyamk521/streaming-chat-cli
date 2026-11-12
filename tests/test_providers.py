"""Tests for LLM provider implementations and factory."""

import json
from types import SimpleNamespace

import httpx
import pytest

from streaming_chat.config import Settings
from streaming_chat.llm import get_llm
from streaming_chat.providers.base import (
    LLMProvider,
    ProviderError,
    StreamChunk,
)
from streaming_chat.providers.groq_provider import GroqProvider
from streaming_chat.providers.ollama_provider import OllamaProvider


def test_groq_provider_streams_text_and_usage() -> None:
    """Verify Groq chunks are normalized into StreamChunk objects."""
    text_chunk = SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content="Hello"),
            )
        ],
        x_groq=None,
    )

    usage_chunk = SimpleNamespace(
        choices=[],
        x_groq=SimpleNamespace(
            usage=SimpleNamespace(
                prompt_tokens=10,
                completion_tokens=5,
            )
        ),
    )

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kwargs: iter([text_chunk, usage_chunk]))
        )
    )

    settings = Settings(_env_file=None)
    provider = GroqProvider(settings, client=fake_client)

    messages = [{"role": "user", "content": "Hello"}]
    chunks = list(provider.stream(messages))

    assert chunks == [
        StreamChunk(text="Hello"),
        StreamChunk(
            text="",
            prompt_tokens=10,
            completion_tokens=5,
        ),
    ]


def test_groq_provider_sends_configured_model() -> None:
    """Verify Groq receives the configured model and streaming option."""
    captured_arguments: dict[str, object] = {}

    def fake_create(**kwargs: object) -> list[object]:
        captured_arguments.update(kwargs)
        return []

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create))
    )

    settings = Settings(
        _env_file=None,
        groq_model="openai/gpt-oss-20b",
    )

    provider = GroqProvider(settings, client=fake_client)

    list(provider.stream([{"role": "user", "content": "Hello"}]))

    assert captured_arguments["model"] == "openai/gpt-oss-20b"
    assert captured_arguments["stream"] is True


def test_groq_provider_wraps_client_errors() -> None:
    """Verify Groq failures become ProviderError exceptions."""

    def failing_create(**kwargs: object) -> None:
        raise RuntimeError("Simulated API failure")

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=failing_create))
    )

    provider = GroqProvider(
        Settings(_env_file=None),
        client=fake_client,
    )

    with pytest.raises(ProviderError, match="Groq failed"):
        list(provider.stream([{"role": "user", "content": "Hello"}]))


def test_ollama_provider_streams_text_and_usage() -> None:
    """Verify Ollama streaming responses are normalized."""
    response_lines = [
        json.dumps(
            {
                "message": {"content": "Hello"},
                "done": False,
            }
        ),
        json.dumps(
            {
                "message": {"content": ""},
                "done": True,
                "prompt_eval_count": 10,
                "eval_count": 5,
            }
        ),
    ]

    def mock_handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/api/chat"

        payload = json.loads(request.content)

        assert payload["model"] == "qwen2.5:3b"
        assert payload["stream"] is True

        return httpx.Response(
            200,
            content="\n".join(response_lines).encode("utf-8"),
        )

    client = httpx.Client(transport=httpx.MockTransport(mock_handler))

    settings = Settings(
        _env_file=None,
        llm_provider="ollama",
    )

    provider = OllamaProvider(settings, client=client)

    messages = [{"role": "user", "content": "Hello"}]
    chunks = list(provider.stream(messages))

    client.close()

    assert chunks == [
        StreamChunk(text="Hello"),
        StreamChunk(
            text="",
            prompt_tokens=10,
            completion_tokens=5,
        ),
    ]


def test_ollama_provider_wraps_http_errors() -> None:
    """Verify HTTP failures become ProviderError exceptions."""

    def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="Internal server error")

    client = httpx.Client(transport=httpx.MockTransport(mock_handler))

    provider = OllamaProvider(
        Settings(_env_file=None, llm_provider="ollama"),
        client=client,
    )

    with pytest.raises(ProviderError, match="Ollama server"):
        list(provider.stream([{"role": "user", "content": "Hello"}]))

    client.close()


def test_llm_factory_selects_groq_provider() -> None:
    """Verify the factory selects Groq from configuration."""
    settings = Settings(
        _env_file=None,
        llm_provider="groq",
        groq_api_key="test-secret-key",
    )

    provider = get_llm(settings)

    assert isinstance(provider, GroqProvider)
    assert isinstance(provider, LLMProvider)


def test_llm_factory_selects_ollama_provider() -> None:
    """Verify the factory selects Ollama from configuration."""
    settings = Settings(
        _env_file=None,
        llm_provider="ollama",
    )

    provider = get_llm(settings)

    assert isinstance(provider, OllamaProvider)
    assert isinstance(provider, LLMProvider)


def test_llm_factory_rejects_missing_groq_key() -> None:
    """Verify Groq initialization requires credentials."""
    settings = Settings(
        _env_file=None,
        llm_provider="groq",
    )

    with pytest.raises(ValueError, match="GROQ_API_KEY is required"):
        get_llm(settings)
