"""Tests for conversation history management."""

import pytest

from streaming_chat.conversation import (
    DEFAULT_SYSTEM_PROMPT,
    Conversation,
)


def test_conversation_starts_with_default_system_prompt() -> None:
    """Verify a new conversation contains the default system prompt."""
    conversation = Conversation()

    assert conversation.get_messages() == [
        {
            "role": "system",
            "content": DEFAULT_SYSTEM_PROMPT,
        }
    ]


def test_conversation_accepts_custom_system_prompt() -> None:
    """Verify a custom system prompt is stored."""
    conversation = Conversation(system_prompt="You are a Python tutor.")

    assert conversation.get_messages() == [
        {
            "role": "system",
            "content": "You are a Python tutor.",
        }
    ]


def test_add_user_message() -> None:
    """Verify user messages are added correctly."""
    conversation = Conversation()

    conversation.add_user_message("What is Python?")

    messages = conversation.get_messages()

    assert messages[-1] == {
        "role": "user",
        "content": "What is Python?",
    }


def test_add_assistant_message() -> None:
    """Verify assistant messages are added correctly."""
    conversation = Conversation()

    conversation.add_assistant_message("Python is a programming language.")

    messages = conversation.get_messages()

    assert messages[-1] == {
        "role": "assistant",
        "content": "Python is a programming language.",
    }


def test_conversation_preserves_message_order() -> None:
    """Verify messages remain in their original chronological order."""
    conversation = Conversation()

    conversation.add_user_message("What is Python?")
    conversation.add_assistant_message("Python is a programming language.")
    conversation.add_user_message("Show me an example.")

    messages = conversation.get_messages()

    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]


def test_get_messages_returns_independent_copy() -> None:
    """Verify callers cannot modify internal conversation history."""
    conversation = Conversation()

    conversation.add_user_message("Hello")

    messages = conversation.get_messages()
    messages[-1]["content"] = "Modified externally"

    assert conversation.get_messages()[-1]["content"] == "Hello"


def test_clear_retains_system_prompt() -> None:
    """Verify clearing removes chat history but retains the system prompt."""
    conversation = Conversation(system_prompt="You are a helpful tutor.")

    conversation.add_user_message("Hello")
    conversation.add_assistant_message("Hi there!")

    conversation.clear()

    assert conversation.get_messages() == [
        {
            "role": "system",
            "content": "You are a helpful tutor.",
        }
    ]


def test_conversation_rejects_empty_message() -> None:
    """Verify empty messages cannot be added."""
    conversation = Conversation()

    with pytest.raises(ValueError, match="Message content cannot be empty"):
        conversation.add_user_message("   ")


def test_conversation_rejects_empty_system_prompt() -> None:
    """Verify an empty system prompt cannot initialize a conversation."""
    with pytest.raises(ValueError, match="System prompt cannot be empty"):
        Conversation(system_prompt="   ")
