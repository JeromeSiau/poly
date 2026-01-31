"""LLM-powered event matching verifier using Claude."""

import json
import re
from dataclasses import dataclass
from typing import Optional

import anthropic

from config.settings import settings


# System prompt for event matching
SYSTEM_PROMPT = """You are an expert at matching sports and prediction market events across different platforms.

Your task is to determine if two events from different prediction market platforms refer to the SAME underlying real-world event.

When comparing events, consider:
1. Team/participant names (account for nicknames, abbreviations, city names)
2. Event type (Super Bowl, NBA Finals, World Cup, etc.)
3. Time frame or date if mentioned
4. Specific outcome being predicted (win, over/under, etc.)

Two events match if and only if:
- They refer to the exact same real-world event
- They predict the same outcome
- A bet on one is economically equivalent to a bet on the other

Be STRICT about matching. Events must be semantically identical, not just similar.

Examples:
- "Chiefs win Super Bowl LIX" and "Kansas City Chiefs to win Super Bowl LIX" = MATCH
- "Chiefs win Super Bowl LIX" and "Chiefs win Super Bowl LX" = NO MATCH (different years)
- "Lakers win NBA Finals" and "Lakers win 2025 NBA Finals" = ONLY MATCH if context makes it clear both refer to 2025

Respond ONLY with valid JSON in this exact format:
{
    "is_match": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation of your decision"
}"""


@dataclass
class MatchResult:
    """Result of an LLM-based event match verification."""

    is_match: bool
    confidence: float
    reasoning: str

    def is_high_confidence(self, threshold: float = 0.95) -> bool:
        """Check if this result has high confidence.

        Args:
            threshold: The confidence threshold (default 0.95).

        Returns:
            True if confidence is at or above the threshold.
        """
        return self.confidence >= threshold


class LLMVerifier:
    """Verifies event matches using Claude LLM."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """Initialize the LLM verifier.

        Args:
            api_key: Anthropic API key. Uses settings if not provided.
            model: Model to use. Uses settings if not provided.
        """
        self._api_key = api_key or settings.ANTHROPIC_API_KEY
        self._model = model or settings.LLM_MODEL
        self._client = anthropic.AsyncAnthropic(api_key=self._api_key)
        self._cache: dict[str, MatchResult] = {}

    def _make_cache_key(
        self,
        event1_name: str,
        event1_platform: str,
        event2_name: str,
        event2_platform: str,
    ) -> str:
        """Create a cache key for the event pair.

        Args:
            event1_name: Name of the first event.
            event1_platform: Platform of the first event.
            event2_name: Name of the second event.
            event2_platform: Platform of the second event.

        Returns:
            A cache key string.
        """
        # Sort to ensure consistent key regardless of order
        pair = sorted([
            f"{event1_platform}:{event1_name}",
            f"{event2_platform}:{event2_name}",
        ])
        return "|".join(pair)

    def _parse_response(self, response_text: str) -> MatchResult:
        """Parse the LLM response into a MatchResult.

        Handles JSON both with and without markdown code blocks.

        Args:
            response_text: The raw response text from the LLM.

        Returns:
            A MatchResult object.

        Raises:
            ValueError: If the response cannot be parsed.
        """
        # Try to extract JSON from markdown code blocks
        json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1).strip()
        else:
            json_str = response_text.strip()

        try:
            data = json.loads(json_str)
            return MatchResult(
                is_match=data["is_match"],
                confidence=float(data["confidence"]),
                reasoning=data.get("reasoning", ""),
            )
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            raise ValueError(f"Failed to parse LLM response: {response_text}") from e

    async def verify_match(
        self,
        event1_name: str,
        event1_platform: str,
        event2_name: str,
        event2_platform: str,
        additional_context: Optional[str] = None,
    ) -> MatchResult:
        """Verify if two events are the same using LLM.

        Args:
            event1_name: Name of the first event.
            event1_platform: Platform of the first event (e.g., 'polymarket').
            event2_name: Name of the second event.
            event2_platform: Platform of the second event (e.g., 'overtime').
            additional_context: Optional additional context for matching.

        Returns:
            A MatchResult indicating if the events match.
        """
        # Check cache first
        cache_key = self._make_cache_key(
            event1_name, event1_platform, event2_name, event2_platform
        )
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Build the user prompt
        prompt = f"""Compare these two prediction market events:

Event 1 (from {event1_platform}):
"{event1_name}"

Event 2 (from {event2_platform}):
"{event2_name}"
"""
        if additional_context:
            prompt += f"\nAdditional context:\n{additional_context}\n"

        prompt += "\nAre these the same event? Respond with JSON only."

        # Call Claude API
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=256,
            system=SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": prompt}
            ],
        )

        # Parse response
        response_text = response.content[0].text
        result = self._parse_response(response_text)

        # Cache the result
        self._cache[cache_key] = result

        return result

    def clear_cache(self) -> None:
        """Clear the match result cache."""
        self._cache.clear()
