"""LLM-powered market screener for Polymarket alpha opportunities.

Strategy from MindshareXBT (strat #3):
1. Scan all active Polymarket markets
2. Score each market for alpha potential (volume, uncertainty, category)
3. Send top N markets to LLM (Claude/Perplexity) for deep research
4. Generate trading thesis with confidence level
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

import structlog

logger = structlog.get_logger()

ALPHA_CATEGORIES = {
    "politics": 1.3, "elections": 1.3, "sports": 1.1,
    "crypto": 0.9, "science": 1.2, "entertainment": 0.8,
}


@dataclass
class ScreenedMarket:
    market_id: str
    title: str
    volume_24h: float
    liquidity: float
    price_yes: float
    category: str
    end_date: str
    alpha_score: float
    llm_thesis: Optional[str] = None
    llm_confidence: Optional[float] = None

    @property
    def is_interesting(self) -> bool:
        return self.alpha_score >= 0.6


class MarketScreener:
    def __init__(self, min_alpha_score: float = 0.6):
        self.min_alpha_score = min_alpha_score

    def compute_alpha_score(
        self, volume_24h: float, liquidity: float, price_yes: float,
        category: str, hours_to_resolution: float,
    ) -> float:
        score = 0.0

        if volume_24h > 100000:
            score += 0.30
        elif volume_24h > 50000:
            score += 0.20
        elif volume_24h > 10000:
            score += 0.10

        uncertainty = 1.0 - abs(price_yes - 0.5) * 2
        score += uncertainty * 0.30

        cat_mult = ALPHA_CATEGORIES.get(category, 1.0)
        score *= cat_mult

        if 24 <= hours_to_resolution <= 168:
            score += 0.10
        elif hours_to_resolution < 2:
            score -= 0.10
        elif hours_to_resolution > 720:
            score -= 0.05

        if liquidity > 50000:
            score += 0.05

        return max(0.0, min(1.0, score))

    def screen_markets(self, markets: list[dict[str, Any]]) -> list[ScreenedMarket]:
        results = []
        for market in markets:
            tokens = market.get("tokens", [])
            price_yes = 0.5
            for token in tokens:
                if token.get("outcome") == "Yes":
                    price_yes = token.get("price", 0.5)
                    break

            end_date = market.get("end_date", "")
            hours = 48.0
            if end_date:
                try:
                    end_dt = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
                    hours = max(0, (end_dt - datetime.now(end_dt.tzinfo)).total_seconds() / 3600)
                except (ValueError, TypeError):
                    pass

            score = self.compute_alpha_score(
                volume_24h=market.get("volume_24h", 0),
                liquidity=market.get("liquidity", 0),
                price_yes=price_yes,
                category=market.get("category", "other"),
                hours_to_resolution=hours,
            )

            results.append(ScreenedMarket(
                market_id=market.get("condition_id", ""),
                title=market.get("title", ""),
                volume_24h=market.get("volume_24h", 0),
                liquidity=market.get("liquidity", 0),
                price_yes=price_yes,
                category=market.get("category", "other"),
                end_date=end_date,
                alpha_score=score,
            ))

        results.sort(key=lambda m: m.alpha_score, reverse=True)
        return results
