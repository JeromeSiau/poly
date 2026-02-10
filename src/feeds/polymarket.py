# src/feeds/polymarket.py
"""Polymarket CLOB WebSocket feed for real-time order book updates.

Uses the CLOB Market channel at wss://ws-subscriptions-clob.polymarket.com/ws/market.
Subscribes by sending ``{"assets_ids": [...], "type": "MARKET"}`` with token IDs.

On subscribe, receives an initial ``book`` snapshot (full bids/asks per token),
then incremental ``price_change`` messages with per-level updates.

Public lookup methods accept (condition_id, outcome) pairs — the feed maps them to
token IDs internally via the ``token_map`` passed to ``subscribe_market()``.
"""

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

import structlog
import websockets

logger = structlog.get_logger()
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
        event_type = raw_data.get("event_type", raw_data.get("type", "unknown"))
        market_id = raw_data.get("market_id", raw_data.get("asset_id", ""))
        outcome = raw_data.get("outcome", "")
        price = float(raw_data.get("price", raw_data.get("last_trade_price", 0.0) or 0))
        timestamp = raw_data.get("timestamp", datetime.now().timestamp())

        data: dict[str, Any] = {
            "market_id": market_id,
            "outcome": outcome,
            "price": price,
        }

        if event_type == "trade":
            data["size"] = raw_data.get("size", 0.0)
            data["side"] = raw_data.get("side", "")
        elif event_type in ("book", "book_update", "agg_orderbook"):
            data["bids"] = raw_data.get("bids", [])
            data["asks"] = raw_data.get("asks", [])

        return cls(
            source="polymarket",
            event_type=event_type,
            game="prediction",
            data=data,
            timestamp=float(timestamp) if timestamp else datetime.now().timestamp(),
            match_id=market_id,
            market_id=market_id,
            outcome=outcome,
            price=price,
        )


class PolymarketFeed(BaseFeed):
    """Real-time order book feed from Polymarket CLOB WebSocket.

    Connects to the Market channel, maintains a local copy of the order book
    keyed by token_id, and provides fast price lookups via (condition_id, outcome).
    """

    WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    PING_INTERVAL = 10.0  # seconds

    def __init__(self) -> None:
        super().__init__()
        self._ws: Optional[ClientConnection] = None
        # Orderbook keyed by token_id -> {"bids": [(price,size)...], "asks": [...]}
        self._local_orderbook: dict[str, dict[str, list[tuple[float, float]]]] = {}
        self._subscribed_tokens: set[str] = set()
        # (condition_id, outcome) -> token_id  for backward-compat lookup
        self._token_map: dict[tuple[str, str], str] = {}
        self._keepalive_task: Optional[asyncio.Task] = None
        self._receive_task: Optional[asyncio.Task] = None
        # Pre-computed best levels cache: token_id -> (best_bid, bid_sz, best_ask, ask_sz)
        self._best_cache: dict[str, tuple[Optional[float], Optional[float], Optional[float], Optional[float]]] = {}
        # Event signaled on every book update (for event-driven consumers).
        self.book_updated: asyncio.Event = asyncio.Event()

    async def connect(self) -> None:
        """Establish WebSocket connection to Polymarket CLOB Market channel."""
        if self._connected:
            return

        self._ws = await websockets.connect(self.WS_URL)
        self._connected = True

        self._keepalive_task = asyncio.create_task(self._keepalive_loop())
        self._receive_task = asyncio.create_task(self._receive_loop())

    async def disconnect(self) -> None:
        """Close WebSocket connection."""
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

        if self._ws:
            await self._ws.close()
            self._ws = None

        self._connected = False
        self._subscribed_tokens.clear()
        self._token_map.clear()
        self._local_orderbook.clear()
        self._best_cache.clear()

    async def subscribe(self, game: str, match_id: str) -> None:
        """Subscribe to events for a specific match/market (BaseFeed interface).

        For the CLOB WS you should prefer ``subscribe_market()`` with a token_map.
        """
        await self.subscribe_market(match_id)

    async def unsubscribe(self, game: str, match_id: str) -> None:
        """Unsubscribe from a market (BaseFeed interface)."""
        await self.unsubscribe_market(match_id)

    async def subscribe_market(
        self,
        market_id: str,
        token_map: dict[str, str] | None = None,
    ) -> None:
        """Subscribe to order book updates for a market.

        Args:
            market_id: Condition ID (e.g. "0x123abc…")
            token_map: ``{outcome: token_id}`` mapping.  Required for the CLOB WS
                       to know which token streams to open.  If ``None``, the call
                       is a no-op.
        """
        if not self._ws or not self._connected:
            raise RuntimeError("Not connected. Call connect() first.")

        if not token_map:
            logger.warning("subscribe_market_no_tokens", market_id=market_id)
            return

        new_tokens: list[str] = []
        for outcome, token_id in token_map.items():
            self._token_map[(market_id, outcome)] = token_id
            if token_id not in self._subscribed_tokens:
                new_tokens.append(token_id)
                self._subscribed_tokens.add(token_id)
                self._local_orderbook.setdefault(token_id, {"bids": [], "asks": []})

        if new_tokens:
            msg = json.dumps({
                "assets_ids": new_tokens,
                "type": "MARKET",
            })
            await self._ws.send(msg)
            logger.debug(
                "clob_ws_subscribed",
                market_id=market_id,
                tokens=len(new_tokens),
            )

        self._subscriptions.add(("prediction", market_id))

    async def unsubscribe_market(self, market_id: str) -> None:
        """Unsubscribe from order book updates for a market."""
        to_remove = [(k, v) for k, v in self._token_map.items() if k[0] == market_id]
        tokens_to_unsub = []
        for key, token_id in to_remove:
            del self._token_map[key]
            self._subscribed_tokens.discard(token_id)
            self._local_orderbook.pop(token_id, None)
            self._best_cache.pop(token_id, None)
            tokens_to_unsub.append(token_id)

        if tokens_to_unsub and self._ws and self._connected:
            try:
                msg = json.dumps({
                    "assets_ids": tokens_to_unsub,
                    "operation": "unsubscribe",
                })
                await self._ws.send(msg)
            except Exception:
                pass  # best-effort

        self._subscriptions.discard(("prediction", market_id))

    # ------------------------------------------------------------------
    # Price queries
    # ------------------------------------------------------------------

    def get_best_prices(
        self, market_id: str, outcome: str
    ) -> tuple[Optional[float], Optional[float]]:
        """Get best bid and ask prices for a market outcome.

        Args:
            market_id: Condition ID
            outcome: e.g. "Up", "Down", "YES", "NO"

        Returns:
            Tuple of (best_bid, best_ask). Returns (None, None) if not available.
        """
        token_id = self._token_map.get((market_id, outcome))
        if token_id is None:
            return (None, None)
        cached = self._best_cache.get(token_id)
        if cached is None:
            return (None, None)
        return (cached[0], cached[2])

    def get_best_levels(
        self, market_id: str, outcome: str
    ) -> tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
        """Get best bid/ask prices and sizes for a market outcome.

        Args:
            market_id: Condition ID
            outcome: e.g. "Up", "Down", "YES", "NO"

        Returns:
            Tuple of (best_bid, bid_size, best_ask, ask_size).
        """
        token_id = self._token_map.get((market_id, outcome))
        if token_id is None:
            return (None, None, None, None)
        return self._best_cache.get(token_id, (None, None, None, None))

    def calculate_implied_probability(
        self, best_bid: Optional[float], best_ask: Optional[float]
    ) -> Optional[float]:
        """Calculate implied probability from best bid and ask (mid-price)."""
        if best_bid is None or best_ask is None:
            return None
        return (best_bid + best_ask) / 2

    def get_market_prices(
        self, market_id: str
    ) -> dict[str, tuple[Optional[float], Optional[float]]]:
        """Get prices for all outcomes in a market.

        Returns:
            Dict mapping outcome to (best_bid, best_ask) tuples
        """
        result: dict[str, tuple[Optional[float], Optional[float]]] = {}
        for (cid, outcome) in self._token_map:
            if cid == market_id:
                result[outcome] = self.get_best_prices(market_id, outcome)
        return result

    # ------------------------------------------------------------------
    # Internal: WebSocket loops
    # ------------------------------------------------------------------

    async def _keepalive_loop(self) -> None:
        """Send periodic ping to keep the connection alive."""
        while self._connected and self._ws:
            try:
                await asyncio.sleep(self.PING_INTERVAL)
                if self._ws:
                    await self._ws.ping()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("keepalive_error", error=str(e))
                break

    async def _receive_loop(self) -> None:
        """Receive and process incoming WebSocket messages."""
        while self._connected and self._ws:
            try:
                message = await self._ws.recv()
                if not isinstance(message, str) or not message.strip():
                    continue  # skip binary or empty frames
                if message.strip() in ("pong", "ping"):
                    continue  # skip keepalive responses
                data = json.loads(message)
                await self._process_message(data)
            except asyncio.CancelledError:
                break
            except websockets.exceptions.ConnectionClosed:
                self._connected = False
                break
            except json.JSONDecodeError:
                continue  # skip non-JSON messages
            except Exception as e:
                logger.error("receive_loop_error", error=str(e))
                continue

    async def _process_message(self, data: Any) -> None:
        """Process an incoming CLOB WS message.

        Messages are either:
        - list: initial ``book`` snapshots (one per subscribed token)
        - dict: incremental ``price_change`` events
        """
        if isinstance(data, list):
            for item in data:
                event_type = item.get("event_type", "")
                if event_type == "book":
                    self._handle_book_snapshot(item)
        elif isinstance(data, dict):
            event_type = data.get("event_type", "")
            if event_type == "price_change":
                self._handle_price_change(data)

    def _handle_book_snapshot(self, item: dict[str, Any]) -> None:
        """Handle an initial full orderbook snapshot for a single token.

        Payload: {"event_type":"book","asset_id":"...","bids":[{"price":"0.49","size":"100"}],...}
        """
        token_id = str(item.get("asset_id", ""))
        if not token_id or token_id not in self._subscribed_tokens:
            return

        bids = [
            (float(b["price"]), float(b["size"]))
            for b in item.get("bids", [])
            if float(b.get("size", 0)) > 0
        ]
        asks = [
            (float(a["price"]), float(a["size"]))
            for a in item.get("asks", [])
            if float(a.get("size", 0)) > 0
        ]
        # Sort: bids descending (best first), asks ascending (best first)
        bids.sort(key=lambda x: x[0], reverse=True)
        asks.sort(key=lambda x: x[0])

        self._local_orderbook[token_id] = {"bids": bids, "asks": asks}
        self._refresh_best_cache(token_id, bids, asks)

    def _handle_price_change(self, data: dict[str, Any]) -> None:
        """Handle incremental price level changes.

        Payload: {"event_type":"price_change","market":"0x...","price_changes":[
            {"asset_id":"...","price":"0.49","size":"100","side":"BUY","best_bid":"0.49","best_ask":"0.51"},
        ]}
        """
        for change in data.get("price_changes", []):
            token_id = str(change.get("asset_id", ""))
            if not token_id or token_id not in self._local_orderbook:
                continue

            price = float(change.get("price", 0))
            size = float(change.get("size", 0))
            side = change.get("side", "").upper()

            book = self._local_orderbook[token_id]
            if side == "BUY":
                book["bids"] = self._merge_levels(
                    book["bids"], [(price, size)], descending=True
                )
            elif side == "SELL":
                book["asks"] = self._merge_levels(
                    book["asks"], [(price, size)], descending=False
                )
            self._refresh_best_cache(token_id, book["bids"], book["asks"])

    def _refresh_best_cache(
        self,
        token_id: str,
        bids: list[tuple[float, float]],
        asks: list[tuple[float, float]],
    ) -> None:
        """Update the pre-computed best-levels cache and signal consumers."""
        self._best_cache[token_id] = (
            bids[0][0] if bids else None,
            bids[0][1] if bids else None,
            asks[0][0] if asks else None,
            asks[0][1] if asks else None,
        )
        self.book_updated.set()

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
        levels = {price: size for price, size in existing}

        for price, size in updates:
            if size == 0:
                levels.pop(price, None)
            else:
                levels[price] = size

        result = [(price, size) for price, size in levels.items()]
        result.sort(key=lambda x: x[0], reverse=descending)
        return result
