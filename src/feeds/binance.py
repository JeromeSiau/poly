"""Binance WebSocket feed for real-time CEX price data.

Connects to Binance's trade stream to track real-time prices for
BTC, ETH, SOL, etc. Used by the Crypto Reality Arb engine to detect
price movements before Polymarket odds update (~30s lag).
"""

import asyncio
import json
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Optional

import structlog
import websockets
from websockets import ClientConnection

from .base import BaseFeed, FeedEvent

logger = structlog.get_logger()

BINANCE_WS_BASE = "wss://stream.binance.com:9443/ws"


@dataclass
class BinanceTick:
    """A single trade tick from Binance."""

    symbol: str
    price: float
    quantity: float
    timestamp: float  # Unix seconds

    @classmethod
    def from_raw(cls, raw: dict[str, Any]) -> "BinanceTick":
        return cls(
            symbol=raw.get("s", ""),
            price=float(raw.get("p", 0.0)),
            quantity=float(raw.get("q", 0.0)),
            timestamp=float(raw.get("T", 0)) / 1000.0,
        )


class BinanceFeed(BaseFeed):
    """Real-time trade feed from Binance WebSocket.

    Maintains a rolling window of recent trades per symbol
    to compute VWAP-based fair value and price direction.
    """

    def __init__(
        self,
        symbols: Optional[list[str]] = None,
        fair_value_window: int = 10,
    ):
        super().__init__()
        self.symbols = symbols or ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        self._fair_value_window = fair_value_window
        self._recent_trades: dict[str, deque[BinanceTick]] = {}
        self._ws: Optional[ClientConnection] = None
        self._direction_threshold: float = 0.001  # 0.1% move = directional

    async def connect(self) -> None:
        streams = "/".join(s.lower() + "@trade" for s in self.symbols)
        url = f"{BINANCE_WS_BASE}/{streams}"
        self._ws = await websockets.connect(url)
        self._connected = True
        logger.info("binance_feed_connected", symbols=self.symbols)

    async def disconnect(self) -> None:
        if self._ws:
            await self._ws.close()
        self._connected = False

    async def subscribe(self, game: str, match_id: str) -> None:
        pass

    async def listen(self) -> None:
        """Listen for trade events and update internal state."""
        if not self._ws:
            raise RuntimeError("Not connected")

        try:
            async for message in self._ws:
                try:
                    data = json.loads(message)
                except json.JSONDecodeError:
                    logger.error("binance_json_decode_error", message=message[:200])
                    continue

                if "data" in data:
                    data = data["data"]

                tick = BinanceTick.from_raw(data)
                if not tick.symbol:
                    continue

                if tick.symbol not in self._recent_trades:
                    self._recent_trades[tick.symbol] = deque(maxlen=self._fair_value_window)
                self._recent_trades[tick.symbol].append(tick)

                event = FeedEvent(
                    source="binance",
                    event_type="trade",
                    game="crypto",
                    data={"symbol": tick.symbol, "price": tick.price, "qty": tick.quantity},
                    timestamp=tick.timestamp,
                    match_id=tick.symbol,
                )
                await self._emit(event)
        except Exception as exc:
            logger.error("binance_websocket_error", error=str(exc))
            self._connected = False
            raise

    def get_recent_trades(self, symbol: str) -> list[BinanceTick]:
        """Get recent trades for a symbol."""
        return list(self._recent_trades.get(symbol, []))

    def get_fair_value(self, symbol: str) -> Optional[float]:
        """Calculate VWAP fair value from recent trades."""
        trades = self._recent_trades.get(symbol, [])
        if not trades:
            return None

        total_value = sum(t.price * t.quantity for t in trades)
        total_volume = sum(t.quantity for t in trades)
        if total_volume == 0:
            return None

        return total_value / total_volume

    def get_price_direction(self, symbol: str) -> str:
        """Determine current price direction: UP, DOWN, or NEUTRAL."""
        trades = self._recent_trades.get(symbol, [])
        if len(trades) < 2:
            return "NEUTRAL"

        vwap = self.get_fair_value(symbol)
        if vwap is None or vwap == 0:
            return "NEUTRAL"

        latest_price = trades[-1].price
        pct_move = (latest_price - vwap) / vwap

        if pct_move > self._direction_threshold:
            return "UP"
        elif pct_move < -self._direction_threshold:
            return "DOWN"
        return "NEUTRAL"
