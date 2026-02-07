"""Logical dependency detector for cross-market combinatorial arbitrage.

Based on Kroer et al. 2016 and Saguillo et al. 2025 (strat #5 RohOnChain):
Markets on Polymarket are priced independently but may have logical dependencies.
"""

import json
from dataclasses import dataclass
from typing import Any, Optional

import httpx
import structlog

from config.settings import settings

logger = structlog.get_logger()


@dataclass
class MarketDependency:
    market_a_id: str
    market_a_title: str
    market_b_id: str
    market_b_title: str
    dependency_type: str
    description: str
    valid_outcomes: list[tuple[str, str]]
    invalid_outcomes: list[tuple[str, str]]
    confidence: float

    @property
    def has_dependency(self) -> bool:
        return len(self.invalid_outcomes) > 0

    @property
    def n_valid_outcomes(self) -> int:
        return len(self.valid_outcomes)

    @property
    def n_total_outcomes(self) -> int:
        return len(self.valid_outcomes) + len(self.invalid_outcomes)


class DependencyDetector:
    def __init__(self, confidence_threshold: float = 0.9):
        self.confidence_threshold = confidence_threshold
        self._cache: dict[tuple[str, str], Optional[MarketDependency]] = {}

    def check_single_market_arbitrage(
        self, yes_price: float, no_price: float, min_profit: float = 0.02
    ) -> Optional[dict[str, Any]]:
        total = yes_price + no_price
        if total < (1.0 - min_profit):
            return {"type": "buy_both", "yes_price": yes_price, "no_price": no_price, "total_cost": total, "profit_per_dollar": 1.0 - total}
        elif total > (1.0 + min_profit):
            return {"type": "sell_both", "yes_price": yes_price, "no_price": no_price, "total_received": total, "profit_per_dollar": total - 1.0}
        return None

    def check_pair_arbitrage(
        self, dependency: MarketDependency, prices: dict[str, dict[str, float]], min_profit: float = 0.05,
    ) -> Optional[dict[str, Any]]:
        if not dependency.has_dependency:
            return None
        a_prices = prices.get(dependency.market_a_id, {})
        b_prices = prices.get(dependency.market_b_id, {})
        if not a_prices or not b_prices:
            return None

        if dependency.dependency_type == "implication":
            p_a = a_prices.get("YES", 0.5)
            p_b = b_prices.get("YES", 0.5)
            if p_b > p_a + min_profit:
                return {"type": "implication_violation", "action": "buy_A_sell_B", "market_a": dependency.market_a_id, "market_b": dependency.market_b_id, "p_a": p_a, "p_b": p_b, "profit_estimate": p_b - p_a}

        if dependency.dependency_type == "mutual_exclusion":
            p_a = a_prices.get("YES", 0.5)
            p_b = b_prices.get("YES", 0.5)
            if p_a + p_b > 1.0 + min_profit:
                return {"type": "mutual_exclusion_violation", "action": "sell_both", "market_a": dependency.market_a_id, "market_b": dependency.market_b_id, "p_a": p_a, "p_b": p_b, "profit_estimate": p_a + p_b - 1.0}

        return None

    async def _ask_llm_dependency(self, title_a: str, title_b: str) -> dict[str, Any]:
        prompt = f"""Analyze the logical relationship between these two Polymarket prediction markets:

Market A: "{title_a}"
Market B: "{title_b}"

Determine if there is a logical dependency (implication, mutual exclusion, or correlation).

Respond with JSON:
{{
    "dependency_type": "implication" | "mutual_exclusion" | "correlation" | "independent",
    "description": "Explanation of the dependency",
    "valid_outcomes": [["YES","YES"], ["YES","NO"], ...],
    "invalid_outcomes": [["NO","YES"], ...],
    "confidence": 0.0 to 1.0
}}
Respond ONLY with JSON."""

        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": settings.ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                json={"model": settings.LLM_MODEL, "max_tokens": 1000, "messages": [{"role": "user", "content": prompt}]},
                timeout=30.0,
            )
            response.raise_for_status()
            data = response.json()
            text = data["content"][0]["text"]
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                import re
                match = re.search(r"\{.*\}", text, re.DOTALL)
                if match:
                    return json.loads(match.group())
                logger.warning("llm_invalid_json", text=text[:200])
                return {"dependency_type": "independent", "confidence": 0.0}

    async def detect_dependency(
        self, market_a_id: str, market_a_title: str, market_b_id: str, market_b_title: str,
    ) -> Optional[MarketDependency]:
        cache_key = (market_a_id, market_b_id)
        if cache_key in self._cache:
            return self._cache[cache_key]

        result = await self._ask_llm_dependency(market_a_title, market_b_title)

        if result.get("dependency_type") == "independent":
            self._cache[cache_key] = None
            return None

        if result.get("confidence", 0) < self.confidence_threshold:
            self._cache[cache_key] = None
            return None

        dep = MarketDependency(
            market_a_id=market_a_id, market_a_title=market_a_title,
            market_b_id=market_b_id, market_b_title=market_b_title,
            dependency_type=result["dependency_type"],
            description=result.get("description", ""),
            valid_outcomes=[tuple(o) for o in result.get("valid_outcomes", [])],
            invalid_outcomes=[tuple(o) for o in result.get("invalid_outcomes", [])],
            confidence=result.get("confidence", 0.0),
        )
        self._cache[cache_key] = dep
        return dep
