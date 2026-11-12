"""Ollama LLM provider implementation."""

import json
import logging
from collections.abc import Iterator
from typing import Any

import httpx

from streaming_chat.config import Settings
from streaming_chat.providers.base import (
    ChatMessage,
    LLMProvider,
    ProviderError,
    StreamChunk,
)

logger = logging.getLogger("streaming_chat.providers.ollama")


class OllamaProvider(LLMProvider):
    """LLM provider using a locally hosted Ollama instance."""

    def __init__(
        self,
        settings: Settings,
        client: httpx.Client | None = None,
    ) -> None:
        """Initialize the Ollama provider.

        Args:
            settings: Validated application settings.
            client: Optional injected HTTPX client for testing.
        """
        self._settings = settings
        self._client = client

    def stream(self, messages: list[ChatMessage]) -> Iterator[StreamChunk]:
        """Stream response chunks from Ollama.

        Args:
            messages: Conversation history in standard chat format.

        Yields:
            Normalized response chunks containing text and optional usage.

        Raises:
            ProviderError: If Ollama is unavailable or returns invalid data.
        """
        endpoint = f"{self._settings.ollama_base_url}/api/chat"

        payload = {
            "model": self._settings.ollama_model,
            "messages": messages,
            "stream": True,
        }

        client = self._client or httpx.Client(timeout=60.0)

        try:
            with client.stream("POST", endpoint, json=payload) as response:
                response.raise_for_status()

                for line in response.iter_lines():
                    if not line:
                        continue

                    try:
                        data: dict[str, Any] = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ProviderError("Ollama returned invalid JSON.") from exc

                    if "error" in data:
                        raise ProviderError(f"Ollama returned an error: {data['error']}")

                    message = data.get("message", {})
                    text = message.get("content", "") or ""

                    prompt_tokens = data.get("prompt_eval_count")
                    completion_tokens = data.get("eval_count")

                    if text or prompt_tokens is not None or completion_tokens is not None:
                        yield StreamChunk(
                            text=text,
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                        )

        except httpx.HTTPError as exc:
            logger.exception("Ollama HTTP request failed")
            raise ProviderError("Could not communicate with the Ollama server.") from exc

        except ProviderError:
            logger.exception("Ollama streaming request failed")
            raise

        except Exception as exc:
            logger.exception("Unexpected Ollama streaming error")
            raise ProviderError("Ollama failed to generate a response.") from exc

        finally:
            if self._client is None:
                client.close()
