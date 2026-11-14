"""Conversation history and message management."""

from streaming_chat.providers.base import ChatMessage

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful AI assistant. "
    "Answer clearly, accurately, and concisely. "
    "When you are unsure, acknowledge the uncertainty."
)


class Conversation:
    """Manage messages exchanged between a user and an AI assistant."""

    def __init__(
        self,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> None:
        """Initialize a conversation with a system prompt."""
        if not system_prompt.strip():
            raise ValueError("System prompt cannot be empty.")

        self._system_prompt = system_prompt
        self._messages: list[ChatMessage] = [
            {
                "role": "system",
                "content": system_prompt,
            }
        ]

    def add_user_message(self, content: str) -> None:
        """Add a user message to the conversation."""
        self._add_message("user", content)

    def add_assistant_message(self, content: str) -> None:
        """Add an assistant response to the conversation."""
        self._add_message("assistant", content)

    def _add_message(
        self,
        role: str,
        content: str,
    ) -> None:
        """Validate and append a message."""
        if not content.strip():
            raise ValueError("Message content cannot be empty.")

        self._messages.append(
            {
                "role": role,
                "content": content,
            }
        )

    def get_messages(self) -> list[ChatMessage]:
        """Return a copy of the complete conversation history."""
        return [message.copy() for message in self._messages]

    def clear(self) -> None:
        """Clear conversation history while retaining the system prompt."""
        self._messages = [
            {
                "role": "system",
                "content": self._system_prompt,
            }
        ]
