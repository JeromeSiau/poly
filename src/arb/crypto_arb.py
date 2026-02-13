"""Crypto Reality Arbitrage Engine.

Exploits the ~30-second lag between CEX price movements (Binance)
and Polymarket 15-minute crypto market odds updates.

Strategy (from Browomo/strat #9):
1. Monitor BTC/ETH/SOL trades on Binance WebSocket
2. Detect confirmed price impulse (direction + magnitude)
3. Calculate fair value for Polymarket YES/NO outcome
4. If Polymarket price hasn't adjusted yet → edge exists → trade

The fair value model: if BTC just moved +0.5% on Binance in the last
few seconds, the probability of "BTC up in next 15 min" is higher
than Polymarket currently shows.
"""

import time
from dataclasses import dataclass
from typing import Any, Optional

import structlog

from config.settings import settings

logger = structlog.get_logger()


@dataclass
class CryptoArbOpportunity:
    """A detected crypto reality arb opportunity."""

    symbol: str
    market_id: str
    token_id: str
    side: str  # BUY or SELL
    outcome: str  # Yes or No
    polymarket_price: float
    fair_value_price: float
    cex_direction: str  # UP or DOWN
    edge_pct: float
    timestamp: float
    available_liquidity: Optional[float] = None

    @property
    def is_valid(self) -> bool:
        min_edge = settings.MIN_EDGE_PCT
        max_edge = settings.ANOMALY_THRESHOLD_PCT
        age = time.time() - self.timestamp
        return (
            self.edge_pct >= min_edge
            and self.edge_pct < max_edge
            and 0 < self.polymarket_price < 1
            and age <= 45.0  # 15-min markets move fast
        )


class CryptoArbEngine:
    """Crypto Reality Arb Engine.

    Detects arbitrage between CEX price movements and Polymarket
    15-minute crypto markets.
    """

    def __init__(
        self,
        binance_feed: Optional[Any] = None,
        polymarket_feed: Optional[Any] = None,
        crypto_mapper: Optional[Any] = None,
        guard: Optional[Any] = None,
        allocated_capital: float = 0.0,
        position_manager: Optional[Any] = None,
    ):
        self.binance_feed = binance_feed
        self.polymarket_feed = polymarket_feed
        self.crypto_mapper = crypto_mapper
        self.guard = guard
        self.allocated_capital = allocated_capital
        self.position_manager = position_manager

        self.min_edge_pct = settings.MIN_EDGE_PCT
        self.stale_seconds = 45.0  # Max age for 15-min market opps
        self.fee_bps = settings.POLYMARKET_FEE_BPS

        # Sensitivity: how much a 1% CEX move shifts fair probability
        # Calibrated from Browomo data: 0.5% BTC move → ~15% prob shift
        self._cex_sensitivity = 30.0  # multiplier: 0.5% move × 30 = 15% prob shift

    def estimate_fair_price(
        self,
        direction: str,
        current_polymarket_price: float,
        cex_pct_move: float,
    ) -> float:
        """Estimate fair Polymarket price given CEX movement.

        Args:
            direction: "UP" or "DOWN"
            current_polymarket_price: Current YES price on Polymarket
            cex_pct_move: Percentage move on CEX (e.g., 0.005 for +0.5%)

        Returns:
            Estimated fair YES price (clamped to [0.01, 0.99])
        """
        prob_shift = abs(cex_pct_move) * self._cex_sensitivity

        if direction == "UP":
            fair = current_polymarket_price + prob_shift
        else:
            fair = current_polymarket_price - prob_shift

        return max(0.01, min(0.99, fair))

    def evaluate_opportunity(self, symbol: str) -> Optional[CryptoArbOpportunity]:
        """Evaluate if a CEX price movement creates an arb opportunity.

        Args:
            symbol: CEX trading pair (e.g., "BTCUSDT")

        Returns:
            CryptoArbOpportunity if edge exists, None otherwise
        """
        if not self.binance_feed:
            return None

        direction = self.binance_feed.get_price_direction(symbol)
        if direction == "NEUTRAL":
            return None

        # Get active Polymarket market for this symbol
        if not self.crypto_mapper:
            return None
        market = self.crypto_mapper.get_active_market(symbol)
        if not market:
            return None

        # Get token for the direction
        token_result = self.crypto_mapper.get_token_for_direction(market, direction)
        if not token_result:
            return None
        token_id, outcome = token_result

        # Get current Polymarket price
        if not self.polymarket_feed:
            return None
        best_bid, best_ask = self.polymarket_feed.get_best_prices(
            market.get("condition_id", ""), token_id
        )
        entry_price = best_ask if best_ask and best_ask > 0 else best_bid or 0.5

        # Calculate CEX percentage move
        trades = self.binance_feed.get_recent_trades(symbol)
        if len(trades) < 2:
            return None
        first_price = trades[0].price
        last_price = trades[-1].price
        if first_price == 0:
            return None
        cex_pct_move = (last_price - first_price) / first_price

        # Estimate fair price
        fair_price = self.estimate_fair_price(direction, entry_price, cex_pct_move)

        # Calculate edge
        edge = fair_price - entry_price
        if self.fee_bps > 0:
            edge -= self.fee_bps / 10000.0

        if edge < self.min_edge_pct:
            return None

        return CryptoArbOpportunity(
            symbol=symbol,
            market_id=market.get("condition_id", ""),
            token_id=token_id,
            side="BUY",
            outcome=outcome,
            polymarket_price=entry_price,
            fair_value_price=fair_price,
            cex_direction=direction,
            edge_pct=edge,
            timestamp=time.time(),
        )
