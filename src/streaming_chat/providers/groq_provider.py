"""Groq LLM provider implementation."""

import logging
from collections.abc import Iterator
from typing import Any

from groq import Groq

from streaming_chat.config import Settings
from streaming_chat.providers.base import (
    ChatMessage,
    LLMProvider,
    ProviderError,
    StreamChunk,
)

logger = logging.getLogger("streaming_chat.providers.groq")


class GroqProvider(LLMProvider):
    """LLM provider using the official Groq Python SDK."""

    def __init__(
        self,
        settings: Settings,
        client: Any | None = None,
    ) -> None:
        """Initialize the Groq provider.

        Args:
            settings: Validated application settings.
            client: Optional injected Groq-compatible client for testing.
        """
        self._settings = settings
        self._client = client or Groq(
            api_key=settings.require_groq_api_key(),
        )

    def stream(self, messages: list[ChatMessage]) -> Iterator[StreamChunk]:
        """Stream response chunks from Groq.

        Args:
            messages: Conversation history in standard chat format.

        Yields:
            Normalized response chunks containing text and optional usage.

        Raises:
            ProviderError: If Groq fails to generate a response.
        """
        try:
            response_stream = self._client.chat.completions.create(
                messages=messages,
                model=self._settings.groq_model,
                stream=True,
            )

            for chunk in response_stream:
                text = ""

                if chunk.choices:
                    text = chunk.choices[0].delta.content or ""

                prompt_tokens = None
                completion_tokens = None

                groq_metadata = getattr(chunk, "x_groq", None)
                usage = getattr(groq_metadata, "usage", None)

                if usage is not None:
                    prompt_tokens = getattr(usage, "prompt_tokens", None)
                    completion_tokens = getattr(usage, "completion_tokens", None)

                if text or prompt_tokens is not None or completion_tokens is not None:
                    yield StreamChunk(
                        text=text,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                    )

        except Exception as exc:
            logger.exception("Groq streaming request failed")
            raise ProviderError("Groq failed to generate a response.") from exc
