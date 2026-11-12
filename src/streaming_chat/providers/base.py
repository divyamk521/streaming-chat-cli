"""Common interfaces and data structures for LLM providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator, Literal, TypedDict


class ChatMessage(TypedDict):
    """A standard chat message."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str


@dataclass(frozen=True, slots=True)
class StreamChunk:
    """A normalized piece of an LLM response.

    Token counts are optional because providers may only report usage
    after the response has finished.
    """

    text: str = ""
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class ProviderError(RuntimeError):
    """Raised when an LLM provider cannot complete a request."""


class LLMProvider(ABC):
    """Abstract interface implemented by every LLM provider."""

    @abstractmethod
    def stream(self, messages: list[ChatMessage]) -> Iterator[StreamChunk]:
        """Stream a response for the supplied conversation messages."""
        raise NotImplementedError
