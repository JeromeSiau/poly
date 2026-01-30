# src/feeds/polymarket.py
"""Polymarket WebSocket feed for real-time order book updates.

Polymarket provides real-time order book updates with ~100ms latency.
CLOB WebSocket: wss://ws-subscriptions-clob.polymarket.com/ws/

This feed maintains a local copy of the order book for fast price lookups.
"""

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional

import websockets
from websockets import ClientConnection

from .base import BaseFeed, FeedEvent


@dataclass
class OrderBookUpdate(FeedEvent):
    """Event from Polymarket order book feed."""

    market_id: str = ""
    outcome: str = ""
    price: float = 0.0

    @classmethod
    def from_raw(cls, raw_data: dict[str, Any]) -> "OrderBookUpdate":
        """Parse raw Polymarket WebSocket message into an OrderBookUpdate.

        Args:
            raw_data: Raw message data from WebSocket

        Returns:
            OrderBookUpdate with normalized data
        """
        event_type = raw_data.get("type", "unknown")
        market_id = raw_data.get("market_id", "")
        outcome = raw_data.get("outcome", "")
        price = raw_data.get("price", 0.0)
        timestamp = raw_data.get("timestamp", datetime.now().timestamp())

        # Build the data dict with extra fields based on event type
        data: dict[str, Any] = {
            "market_id": market_id,
            "outcome": outcome,
            "price": price,
        }

        if event_type == "trade":
            data["size"] = raw_data.get("size", 0.0)
            data["side"] = raw_data.get("side", "")

        elif event_type == "book_update":
            data["bids"] = raw_data.get("bids", [])
            data["asks"] = raw_data.get("asks", [])

        return cls(
            source="polymarket",
            event_type=event_type,
            game="prediction",  # Polymarket is for prediction markets
            data=data,
            timestamp=float(timestamp),
            match_id=market_id,
            market_id=market_id,
            outcome=outcome,
            price=price,
        )


class PolymarketFeed(BaseFeed):
    """Real-time order book feed from Polymarket WebSocket.

    Maintains a local copy of the order book for fast price lookups.
    Provides ~100ms latency for price updates.
    """

    WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/"
    PING_INTERVAL = 30.0  # seconds

    def __init__(self):
        """Initialize Polymarket feed."""
        super().__init__()
        self._ws: Optional[ClientConnection] = None
        self._local_orderbook: dict[str, dict[str, dict[str, list[tuple[float, float]]]]] = {}
        self._subscribed_markets: set[str] = set()
        self._keepalive_task: Optional[asyncio.Task] = None
        self._receive_task: Optional[asyncio.Task] = None

    async def connect(self) -> None:
        """Establish WebSocket connection to Polymarket."""
        if self._connected:
            return

        self._ws = await websockets.connect(self.WS_URL)
        self._connected = True

        # Start background tasks
        self._keepalive_task = asyncio.create_task(self._keepalive_loop())
        self._receive_task = asyncio.create_task(self._receive_loop())

    async def disconnect(self) -> None:
        """Close WebSocket connection."""
        # Cancel background tasks
        if self._keepalive_task:
            self._keepalive_task.cancel()
            try:
                await self._keepalive_task
            except asyncio.CancelledError:
                pass
            self._keepalive_task = None

        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
            self._receive_task = None

        # Close WebSocket
        if self._ws:
            await self._ws.close()
            self._ws = None

        self._connected = False
        self._subscribed_markets.clear()
        self._local_orderbook.clear()

    async def subscribe(self, game: str, match_id: str) -> None:
        """Subscribe to events for a specific match/market.

        This method is required by BaseFeed but for Polymarket,
        use subscribe_market() instead for clearer semantics.

        Args:
            game: Ignored for Polymarket (always "prediction")
            match_id: Market ID to subscribe to
        """
        await self.subscribe_market(match_id)

    async def subscribe_market(self, market_id: str) -> None:
        """Subscribe to order book updates for a market.

        Args:
            market_id: Polymarket market identifier (e.g., "0x123abc")
        """
        if not self._ws or not self._connected:
            raise RuntimeError("Not connected. Call connect() first.")

        message = json.dumps({
            "type": "subscribe",
            "markets": [market_id],
        })
        await self._ws.send(message)
        self._subscribed_markets.add(market_id)
        self._subscriptions.add(("prediction", market_id))

        # Initialize orderbook structure for this market
        if market_id not in self._local_orderbook:
            self._local_orderbook[market_id] = {
                "YES": {"bids": [], "asks": []},
                "NO": {"bids": [], "asks": []},
            }

    async def unsubscribe_market(self, market_id: str) -> None:
        """Unsubscribe from order book updates for a market.

        Args:
            market_id: Polymarket market identifier
        """
        if not self._ws or not self._connected:
            return

        message = json.dumps({
            "type": "unsubscribe",
            "markets": [market_id],
        })
        await self._ws.send(message)
        self._subscribed_markets.discard(market_id)
        self._subscriptions.discard(("prediction", market_id))

        # Remove orderbook data for this market
        self._local_orderbook.pop(market_id, None)

    def get_best_prices(
        self, market_id: str, outcome: str
    ) -> tuple[Optional[float], Optional[float]]:
        """Get best bid and ask prices for a market outcome.

        Args:
            market_id: Polymarket market identifier
            outcome: "YES" or "NO"

        Returns:
            Tuple of (best_bid, best_ask). Returns (None, None) if not available.
        """
        if market_id not in self._local_orderbook:
            return (None, None)

        orderbook = self._local_orderbook[market_id].get(outcome, {})
        bids = orderbook.get("bids", [])
        asks = orderbook.get("asks", [])

        # Bids are sorted descending (highest first), asks ascending (lowest first)
        best_bid = bids[0][0] if bids else None
        best_ask = asks[0][0] if asks else None

        return (best_bid, best_ask)

    def calculate_implied_probability(
        self, best_bid: Optional[float], best_ask: Optional[float]
    ) -> Optional[float]:
        """Calculate implied probability from best bid and ask.

        The implied probability is the mid-price between bid and ask.

        Args:
            best_bid: Best bid price
            best_ask: Best ask price

        Returns:
            Implied probability (mid-price) or None if prices unavailable
        """
        if best_bid is None or best_ask is None:
            return None

        return (best_bid + best_ask) / 2

    def get_market_prices(
        self, market_id: str
    ) -> dict[str, tuple[Optional[float], Optional[float]]]:
        """Get prices for all outcomes in a market.

        Args:
            market_id: Polymarket market identifier

        Returns:
            Dict mapping outcome ("YES", "NO") to (best_bid, best_ask) tuples
        """
        if market_id not in self._local_orderbook:
            return {}

        result = {}
        for outcome in self._local_orderbook[market_id]:
            result[outcome] = self.get_best_prices(market_id, outcome)

        return result

    async def _keepalive_loop(self) -> None:
        """Send periodic ping messages to keep the connection alive."""
        while self._connected and self._ws:
            try:
                await asyncio.sleep(self.PING_INTERVAL)
                if self._ws:
                    await self._ws.ping()
            except asyncio.CancelledError:
                break
            except Exception:
                # Connection may have been lost
                break

    async def _receive_loop(self) -> None:
        """Receive and process incoming WebSocket messages."""
        while self._connected and self._ws:
            try:
                message = await self._ws.recv()
                data = json.loads(message)
                await self._process_message(data)
            except asyncio.CancelledError:
                break
            except websockets.exceptions.ConnectionClosed:
                self._connected = False
                break
            except Exception:
                # Log error but continue processing
                continue

    async def _process_message(self, data: dict[str, Any]) -> None:
        """Process an incoming WebSocket message.

        Args:
            data: Parsed JSON message data
        """
        msg_type = data.get("type", "")

        if msg_type in ("price_change", "trade", "book_update"):
            # Create event and emit to callbacks
            update = OrderBookUpdate.from_raw(data)
            await self._emit(update)

            # Update local orderbook if applicable
            if msg_type == "book_update":
                self._update_orderbook(data)

        elif msg_type == "snapshot":
            # Full orderbook snapshot
            self._handle_snapshot(data)

    def _update_orderbook(self, data: dict[str, Any]) -> None:
        """Update local orderbook with incremental data.

        Args:
            data: Order book update message
        """
        market_id = data.get("market_id", "")
        outcome = data.get("outcome", "")

        if not market_id or not outcome:
            return

        if market_id not in self._local_orderbook:
            self._local_orderbook[market_id] = {
                "YES": {"bids": [], "asks": []},
                "NO": {"bids": [], "asks": []},
            }

        orderbook = self._local_orderbook[market_id].get(outcome, {"bids": [], "asks": []})

        # Update bids
        if "bids" in data:
            new_bids = [(float(b[0]), float(b[1])) for b in data["bids"]]
            orderbook["bids"] = self._merge_levels(orderbook["bids"], new_bids, descending=True)

        # Update asks
        if "asks" in data:
            new_asks = [(float(a[0]), float(a[1])) for a in data["asks"]]
            orderbook["asks"] = self._merge_levels(orderbook["asks"], new_asks, descending=False)

        self._local_orderbook[market_id][outcome] = orderbook

    def _handle_snapshot(self, data: dict[str, Any]) -> None:
        """Handle a full orderbook snapshot.

        Args:
            data: Snapshot message data
        """
        market_id = data.get("market_id", "")
        if not market_id:
            return

        self._local_orderbook[market_id] = {}

        for outcome_data in data.get("outcomes", []):
            outcome = outcome_data.get("outcome", "")
            bids = [(float(b[0]), float(b[1])) for b in outcome_data.get("bids", [])]
            asks = [(float(a[0]), float(a[1])) for a in outcome_data.get("asks", [])]

            # Sort bids descending, asks ascending
            bids.sort(key=lambda x: x[0], reverse=True)
            asks.sort(key=lambda x: x[0])

            self._local_orderbook[market_id][outcome] = {
                "bids": bids,
                "asks": asks,
            }

    def _merge_levels(
        self,
        existing: list[tuple[float, float]],
        updates: list[tuple[float, float]],
        descending: bool = False,
    ) -> list[tuple[float, float]]:
        """Merge price level updates into existing orderbook levels.

        Args:
            existing: Existing price levels [(price, size), ...]
            updates: New/updated price levels
            descending: If True, sort descending (for bids)

        Returns:
            Merged and sorted price levels
        """
        # Convert to dict for easy updates
        levels = {price: size for price, size in existing}

        for price, size in updates:
            if size == 0:
                # Remove level
                levels.pop(price, None)
            else:
                # Add or update level
                levels[price] = size

        # Convert back to sorted list
        result = [(price, size) for price, size in levels.items()]
        result.sort(key=lambda x: x[0], reverse=descending)

        return result
