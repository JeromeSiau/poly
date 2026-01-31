"""Observe and capture Polymarket prices."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import structlog

logger = structlog.get_logger()


@dataclass
class PriceCapture:
    """A captured price at a point in time."""

    market_id: str
    outcome: str
    price: Optional[float]
    timestamp: datetime
    error: Optional[str] = None


@dataclass
class MarketObserver:
    """Observe Polymarket prices and capture at intervals."""

    # Polymarket feed will be injected
    polymarket_feed: Optional[object] = None

    # Default follow-up intervals (seconds)
    default_intervals: list[float] = field(
        default_factory=lambda: [30.0, 60.0, 120.0]
    )

    async def capture_price(
        self,
        market_id: str,
        outcome: str,
    ) -> PriceCapture:
        """Capture current price for a market outcome."""
        try:
            price = await self._fetch_price(market_id, outcome)
            return PriceCapture(
                market_id=market_id,
                outcome=outcome,
                price=price,
                timestamp=datetime.utcnow(),
            )
        except Exception as e:
            logger.error("price_capture_failed", error=str(e), market_id=market_id)
            return PriceCapture(
                market_id=market_id,
                outcome=outcome,
                price=None,
                timestamp=datetime.utcnow(),
                error=str(e),
            )

    async def capture_with_followups(
        self,
        market_id: str,
        outcome: str,
        intervals: Optional[list[float]] = None,
    ) -> list[PriceCapture]:
        """Capture price now and at follow-up intervals."""
        intervals = intervals or self.default_intervals
        captures = []

        # Capture T+0
        capture = await self.capture_price(market_id, outcome)
        captures.append(capture)

        # Schedule follow-ups
        for interval in intervals:
            await asyncio.sleep(interval)
            capture = await self.capture_price(market_id, outcome)
            captures.append(capture)

        return captures

    async def _fetch_price(
        self,
        market_id: str,
        outcome: str,
    ) -> float:
        """Fetch current price from Polymarket."""
        if self.polymarket_feed is None:
            logger.warning("no_polymarket_feed", market_id=market_id)
            return 0.50

        return await self.polymarket_feed.get_price(market_id, outcome)
