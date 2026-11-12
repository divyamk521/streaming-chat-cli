"""Structured JSON logging configuration."""

import json
import logging
from datetime import datetime, timezone
from typing import TextIO


class JsonFormatter(logging.Formatter):
    """Format log records as structured JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        """Convert a log record into a JSON string."""
        log_entry: dict[str, str] = {
            "timestamp": datetime.fromtimestamp(
                record.created,
                tz=timezone.utc,
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, ensure_ascii=False)


def configure_logging(
    level: str = "INFO",
    stream: TextIO | None = None,
) -> None:
    """Configure application-wide JSON logging.

    Args:
        level: Logging severity level.
        stream: Optional output stream, primarily useful for testing.

    Raises:
        ValueError: If the supplied logging level is invalid.
    """
    normalized_level = level.upper()
    numeric_level = logging.getLevelNamesMapping().get(normalized_level)

    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid logging level: {level}")

    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    root_logger.handlers.clear()

    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())

    root_logger.addHandler(handler)
