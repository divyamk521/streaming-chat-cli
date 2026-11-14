"""Application-level chat streaming service."""

from collections.abc import Iterator

from streaming_chat.config import Settings
from streaming_chat.llm import get_llm
from streaming_chat.providers.base import ChatMessage, StreamChunk


def stream_chat(
    messages: list[ChatMessage],
    settings: Settings | None = None,
) -> Iterator[StreamChunk]:
    """Stream a chat response using the configured LLM provider.

    Args:
        messages: Conversation messages in standard chat format.
        settings: Optional application settings.

    Yields:
        Normalized response chunks as they arrive from the provider.

    Raises:
        ValueError: If the conversation contains no messages.
        ProviderError: If the configured provider fails.
    """
    if not messages:
        raise ValueError("At least one chat message is required.")

    provider = get_llm(settings)

    yield from provider.stream(messages)
