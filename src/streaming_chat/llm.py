"""Central entry point for all LLM provider interactions."""

from streaming_chat.config import Settings, get_settings
from streaming_chat.providers.base import LLMProvider
from streaming_chat.providers.groq_provider import GroqProvider
from streaming_chat.providers.ollama_provider import OllamaProvider


def get_llm(settings: Settings | None = None) -> LLMProvider:
    """Create an LLM provider based on application configuration.

    Args:
        settings: Optional application settings. When omitted, settings
            are loaded from the environment.

    Returns:
        An initialized LLM provider.

    Raises:
        ValueError: If the configured provider is unsupported or credentials
            are missing.
    """
    application_settings = settings or get_settings()

    if application_settings.llm_provider == "groq":
        return GroqProvider(application_settings)

    if application_settings.llm_provider == "ollama":
        return OllamaProvider(application_settings)

    raise ValueError(f"Unsupported LLM provider: {application_settings.llm_provider}")
