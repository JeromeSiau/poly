"""Event matching module for cross-market arbitrage."""

from src.matching.event_matcher import CrossMarketMatcher, MatchedEvent
from src.matching.llm_verifier import LLMVerifier, MatchResult
from src.matching.normalizer import EventNormalizer

__all__ = [
    "CrossMarketMatcher",
    "EventNormalizer",
    "LLMVerifier",
    "MatchedEvent",
    "MatchResult",
]
