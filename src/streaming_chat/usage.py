"""Token usage tracking and API cost calculation."""

from dataclasses import dataclass
from math import ceil, isfinite

from streaming_chat.providers.base import ChatMessage, StreamChunk


@dataclass(frozen=True, slots=True)
class CostRates:
    """API pricing in USD per one million tokens."""

    input_per_million: float
    output_per_million: float

    def __post_init__(self) -> None:
        """Reject negative or non-finite pricing values."""
        for name, value in (
            ("input_per_million", self.input_per_million),
            ("output_per_million", self.output_per_million),
        ):
            if not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite, non-negative number.")


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Token counts and calculated cost for one response."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    input_cost_usd: float
    output_cost_usd: float
    total_cost_usd: float
    estimated: bool


class UsageTracker:
    """Collect streamed output and calculate token usage and cost."""

    def __init__(
        self,
        rates: CostRates,
        chars_per_token: int = 4,
    ) -> None:
        """Initialize a tracker for one model response."""
        if chars_per_token < 1:
            raise ValueError("chars_per_token must be at least 1.")

        self._rates = rates
        self._chars_per_token = chars_per_token
        self._response_parts: list[str] = []
        self._reported_prompt_tokens: int | None = None
        self._reported_completion_tokens: int | None = None

    def record_chunk(self, chunk: StreamChunk) -> None:
        """Record response text and any provider-reported usage."""
        self._response_parts.append(chunk.text)

        if chunk.prompt_tokens is not None:
            self._validate_token_count(chunk.prompt_tokens)
            self._reported_prompt_tokens = chunk.prompt_tokens

        if chunk.completion_tokens is not None:
            self._validate_token_count(chunk.completion_tokens)
            self._reported_completion_tokens = chunk.completion_tokens

    def calculate(self, messages: list[ChatMessage]) -> TokenUsage:
        """Calculate token counts and costs for the completed response."""
        prompt_text = "\n".join(message["content"] for message in messages)
        completion_text = "".join(self._response_parts)

        estimated_prompt_tokens = self._estimate_tokens(prompt_text)
        estimated_completion_tokens = self._estimate_tokens(completion_text)

        prompt_tokens = (
            self._reported_prompt_tokens
            if self._reported_prompt_tokens is not None
            else estimated_prompt_tokens
        )
        completion_tokens = (
            self._reported_completion_tokens
            if self._reported_completion_tokens is not None
            else estimated_completion_tokens
        )

        input_cost = (prompt_tokens / 1_000_000) * self._rates.input_per_million
        output_cost = (completion_tokens / 1_000_000) * self._rates.output_per_million

        return TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            input_cost_usd=input_cost,
            output_cost_usd=output_cost,
            total_cost_usd=input_cost + output_cost,
            estimated=(
                self._reported_prompt_tokens is None or self._reported_completion_tokens is None
            ),
        )

    def _estimate_tokens(self, text: str) -> int:
        """Estimate tokens using a simple character-based heuristic."""
        if not text:
            return 0

        return ceil(len(text) / self._chars_per_token)

    @staticmethod
    def _validate_token_count(value: int) -> None:
        """Reject invalid provider-reported token counts."""
        if isinstance(value, bool) or value < 0:
            raise ValueError("Token counts must be non-negative integers.")
