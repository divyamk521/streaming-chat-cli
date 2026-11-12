"""Tests for initial project setup."""

from streaming_chat import __version__


def test_package_version() -> None:
    """Verify that the application exposes its initial version."""
    assert __version__ == "0.1.0"
