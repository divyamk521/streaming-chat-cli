"""Tests for token usage tracking and cost calculation."""

import pytest

from streaming_chat.providers.base import StreamChunk
from streaming_chat.usage import CostRates, UsageTracker


@pytest.fixture
def rates() -> CostRates:
    """Provide predictable pricing for tests."""
    return CostRates(
        input_per_million=1.0,
        output_per_million=2.0,
    )


def test_calculates_provider_reported_usage(rates: CostRates) -> None:
    """Provider-reported token counts should be preferred."""
    tracker = UsageTracker(rates)

    tracker.record_chunk(StreamChunk(text="Hello"))
    tracker.record_chunk(
        StreamChunk(
            text=" world",
            prompt_tokens=100,
            completion_tokens=20,
        )
    )

    usage = tracker.calculate([{"role": "user", "content": "Explain Python."}])

    assert usage.prompt_tokens == 100
    assert usage.completion_tokens == 20
    assert usage.total_tokens == 120
    assert usage.estimated is False


def test_estimates_usage_when_provider_does_not_report_tokens(
    rates: CostRates,
) -> None:
    """Missing provider usage should trigger character-based estimates."""
    tracker = UsageTracker(rates)

    tracker.record_chunk(StreamChunk(text="Hello world"))

    usage = tracker.calculate([{"role": "user", "content": "12345678"}])

    assert usage.prompt_tokens == 2
    assert usage.completion_tokens == 3
    assert usage.total_tokens == 5
    assert usage.estimated is True


def test_estimates_only_missing_token_count(rates: CostRates) -> None:
    """A reported prompt count should not replace an estimated output count."""
    tracker = UsageTracker(rates)

    tracker.record_chunk(
        StreamChunk(
            text="12345678",
            prompt_tokens=50,
        )
    )

    usage = tracker.calculate([{"role": "user", "content": "Question"}])

    assert usage.prompt_tokens == 50
    assert usage.completion_tokens == 2
    assert usage.estimated is True


def test_calculates_input_and_output_costs(rates: CostRates) -> None:
    """Input and output costs should use their respective rates."""
    tracker = UsageTracker(rates)

    tracker.record_chunk(
        StreamChunk(
            text="Answer",
            prompt_tokens=1_000_000,
            completion_tokens=500_000,
        )
    )

    usage = tracker.calculate([{"role": "user", "content": "Question"}])

    assert usage.input_cost_usd == 1.0
    assert usage.output_cost_usd == 1.0
    assert usage.total_cost_usd == 2.0


def test_empty_response_has_zero_completion_tokens(rates: CostRates) -> None:
    """An empty response should not produce estimated output tokens."""
    tracker = UsageTracker(rates)

    usage = tracker.calculate([{"role": "user", "content": "Hello"}])

    assert usage.completion_tokens == 0


def test_multiple_chunks_are_combined(rates: CostRates) -> None:
    """Text from multiple chunks should be counted together."""
    tracker = UsageTracker(rates)

    tracker.record_chunk(StreamChunk(text="1234"))
    tracker.record_chunk(StreamChunk(text="5678"))

    usage = tracker.calculate([{"role": "user", "content": "1234"}])

    assert usage.completion_tokens == 2


def test_rejects_negative_pricing() -> None:
    """Negative API pricing should not be accepted."""
    with pytest.raises(ValueError, match="finite, non-negative"):
        CostRates(
            input_per_million=-1.0,
            output_per_million=2.0,
        )


def test_rejects_non_finite_pricing() -> None:
    """Infinite API pricing should not be accepted."""
    with pytest.raises(ValueError, match="finite, non-negative"):
        CostRates(
            input_per_million=float("inf"),
            output_per_million=2.0,
        )


def test_rejects_invalid_chars_per_token(rates: CostRates) -> None:
    """The estimation ratio must be a positive integer."""
    with pytest.raises(ValueError, match="at least 1"):
        UsageTracker(rates, chars_per_token=0)


def test_rejects_negative_provider_token_count(rates: CostRates) -> None:
    """Negative provider-reported token counts should be rejected."""
    tracker = UsageTracker(rates)

    with pytest.raises(ValueError, match="non-negative integers"):
        tracker.record_chunk(
            StreamChunk(
                text="Hello",
                prompt_tokens=-1,
            )
        )
