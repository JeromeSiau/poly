"""Contrarian NO Bet Scanner.

Identifies Polymarket markets where YES is overpriced due to optimism bias.
Strategy from DidiTrading / NeverYES (strat #4):

NeverYES enters NO at ~0.45-0.55 on hype-driven markets (mainly crypto
FDV/launch bets) where retail overestimates the YES outcome.  The edge
comes from the gap between the market-implied probability and the true
base rate for these events.

Key insight: "treat the market title as marketing, trade the resolution
criteria" — mispricings appear when traders overweight headline hype
versus settlement specifics.

Scoring criteria:
- YES price in optimism-bias sweet spot (0.35-0.65)
- Market category indicates hype-prone topic (FDV, launch, price targets)
- High volume (liquid, can exit anytime)
- Reasonable time to resolution
"""

from dataclasses import dataclass
from typing import Any, Optional

import structlog

from config.settings import settings

logger = structlog.get_logger()

# Hype-prone market keywords — YES tends to be overpriced on these
HYPE_KEYWORDS = [
    "fdv", "fully diluted", "market cap above", "market cap over",
    "price above", "price over", "reach $", "hit $",
    "above $", "over $", "launch above", "ath",
    "all-time high", "all time high",
]
# Moderate hype — still biased but less reliably
MODERATE_HYPE_KEYWORDS = [
    "win", "will pass", "will be approved", "will launch",
    "token", "airdrop", "tge",
]


@dataclass
class NoBetOpportunity:
    """A candidate NO bet opportunity."""

    market_id: str
    token_id: str
    title: str
    yes_price: float
    no_price: float
    estimated_no_probability: float
    edge_pct: float
    volume_24h: float
    liquidity: float

    @property
    def expected_return(self) -> float:
        """Expected return per dollar invested in NO."""
        win_payout = 1.0 - self.no_price
        loss = self.no_price
        p_no = self.estimated_no_probability
        return p_no * win_payout - (1 - p_no) * loss

    @property
    def is_valid(self) -> bool:
        return (
            self.edge_pct > 0.01
            and self.liquidity >= 1000.0
            and self.volume_24h >= 5000.0
            and 0 < self.no_price < 1
        )


class NoBetScanner:
    """Scans Polymarket for contrarian NO bets on hype-driven markets."""

    def __init__(
        self,
        min_yes_price: float = 0.35,
        max_yes_price: float = 0.65,
        min_liquidity: float = 1000.0,
        min_volume_24h: float = 5000.0,
    ):
        self.min_yes_price = min_yes_price
        self.max_yes_price = max_yes_price
        self.min_liquidity = min_liquidity
        self.min_volume_24h = min_volume_24h

    def score_market(
        self,
        title: str,
        yes_price: float,
        volume_24h: float,
        end_date_days: int,
    ) -> float:
        """Score a market for NO bet suitability (0.0 to 1.0).

        Higher scores indicate stronger hype/optimism bias signal.
        """
        score = 0.3  # base score

        # Sweet spot: NeverYES enters at YES ~0.45-0.55
        if 0.45 <= yes_price <= 0.55:
            score += 0.25
        elif 0.40 <= yes_price <= 0.60:
            score += 0.15
        elif 0.35 <= yes_price <= 0.65:
            score += 0.05

        # Hype keyword detection
        title_lower = title.lower()
        if any(kw in title_lower for kw in HYPE_KEYWORDS):
            score += 0.25
        elif any(kw in title_lower for kw in MODERATE_HYPE_KEYWORDS):
            score += 0.10

        # Volume = liquidity = can exit
        if volume_24h > 100000:
            score += 0.10
        elif volume_24h > 50000:
            score += 0.05

        # Shorter resolution = faster payoff
        if end_date_days <= 7:
            score += 0.10
        elif end_date_days <= 30:
            score += 0.05
        elif end_date_days > 180:
            score -= 0.10

        return max(0.0, min(1.0, score))

    def filter_candidates(
        self, markets: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Filter markets in the optimism-bias YES price range."""
        candidates = []
        for market in markets:
            tokens = market.get("tokens", [])
            yes_price = None
            for token in tokens:
                if token.get("outcome") == "Yes":
                    yes_price = token.get("price", 1.0)
                    break

            if (
                yes_price is not None
                and self.min_yes_price <= yes_price <= self.max_yes_price
            ):
                candidates.append(market)

        return candidates
