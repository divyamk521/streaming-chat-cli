"""Tests for structured JSON logging."""

import json
import logging
from io import StringIO

import pytest

from streaming_chat.logging_config import configure_logging


def test_logging_outputs_valid_json() -> None:
    """Verify application logs are valid JSON."""
    output = StringIO()

    configure_logging("INFO", stream=output)

    logger = logging.getLogger("streaming_chat.tests")
    logger.info("Application initialized")

    log_entry = json.loads(output.getvalue())

    assert log_entry["level"] == "INFO"
    assert log_entry["logger"] == "streaming_chat.tests"
    assert log_entry["message"] == "Application initialized"
    assert "timestamp" in log_entry


def test_debug_logs_are_filtered() -> None:
    """Verify DEBUG messages are filtered at INFO level."""
    output = StringIO()

    configure_logging("INFO", stream=output)

    logger = logging.getLogger("streaming_chat.tests")
    logger.debug("This message should not appear")
    logger.info("This message should appear")

    logs = output.getvalue()

    assert "This message should not appear" not in logs
    assert "This message should appear" in logs


def test_debug_level_allows_debug_messages() -> None:
    """Verify DEBUG messages appear when DEBUG logging is enabled."""
    output = StringIO()

    configure_logging("DEBUG", stream=output)

    logger = logging.getLogger("streaming_chat.tests")
    logger.debug("Debug message")

    log_entry = json.loads(output.getvalue())

    assert log_entry["level"] == "DEBUG"
    assert log_entry["message"] == "Debug message"


def test_invalid_logging_level_is_rejected() -> None:
    """Verify invalid logging levels produce an error."""
    with pytest.raises(ValueError, match="Invalid logging level"):
        configure_logging("INVALID")


def test_logging_level_is_case_insensitive() -> None:
    """Verify lowercase logging levels are accepted."""
    output = StringIO()

    configure_logging("debug", stream=output)

    logger = logging.getLogger("streaming_chat.tests")
    logger.debug("Lowercase level works")

    log_entry = json.loads(output.getvalue())

    assert log_entry["level"] == "DEBUG"
