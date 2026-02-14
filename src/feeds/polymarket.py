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
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

try:
    import orjson
    _json_loads = orjson.loads
except ImportError:
    _json_loads = json.loads

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

    Auto-reconnects on connection drops with exponential backoff, preserving
    subscriptions across reconnections.
    """

    WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    PING_INTERVAL = 10.0  # seconds
    RECONNECT_BASE = 1.0  # initial backoff seconds
    RECONNECT_MAX = 30.0  # max backoff seconds

    def __init__(self) -> None:
        super().__init__()
        self._ws: Optional[ClientConnection] = None
        # Orderbook keyed by token_id -> {"bids": [(price,size)...], "asks": [...]}
        self._local_orderbook: dict[str, dict[str, list[tuple[float, float]]]] = {}
        self._subscribed_tokens: set[str] = set()
        # (condition_id, outcome) -> token_id  for backward-compat lookup
        self._token_map: dict[tuple[str, str], str] = {}
        self._connection_task: Optional[asyncio.Task] = None
        # Pre-computed best levels cache: token_id -> (best_bid, bid_sz, best_ask, ask_sz)
        self._best_cache: dict[str, tuple[Optional[float], Optional[float], Optional[float], Optional[float]]] = {}
        # Event signaled on every book update (for event-driven consumers).
        self.book_updated: asyncio.Event = asyncio.Event()
        self._shutdown = False
        # Timestamp of the last book data received (for staleness checks).
        self.last_update_ts: float = 0.0

    async def connect(self) -> None:
        """Start the auto-reconnecting connection loop."""
        if self._connection_task and not self._connection_task.done():
            return
        self._shutdown = False
        self._connection_task = asyncio.create_task(self._connection_loop())

    async def disconnect(self) -> None:
        """Close WebSocket connection and stop reconnecting."""
        self._shutdown = True

        if self._connection_task:
            self._connection_task.cancel()
            try:
                await self._connection_task
            except asyncio.CancelledError:
                pass
            self._connection_task = None

        await self._close_ws()
        self._subscribed_tokens.clear()
        self._token_map.clear()
        self._local_orderbook.clear()
        self._best_cache.clear()

    async def _close_ws(self) -> None:
        """Close the raw WebSocket if open."""
        self._connected = False
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

    async def _connection_loop(self) -> None:
        """Connect, run, and auto-reconnect on failure."""
        backoff = self.RECONNECT_BASE
        while not self._shutdown:
            try:
                self._ws = await websockets.connect(self.WS_URL, proxy=None)
                self._connected = True
                backoff = self.RECONNECT_BASE
                logger.info("polymarket_ws_connected")

                # Re-subscribe all tokens from previous session.
                await self._resubscribe_all()

                # Run keepalive + receive until one fails.
                await asyncio.gather(
                    self._keepalive_loop(),
                    self._receive_loop(),
                )
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("polymarket_ws_disconnected", error=str(exc), reconnect_in=backoff)

            # Clean up stale data before reconnecting.
            await self._close_ws()
            self._local_orderbook.clear()
            self._best_cache.clear()

            if self._shutdown:
                break
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, self.RECONNECT_MAX)

    async def _resubscribe_all(self) -> None:
        """Re-subscribe all previously tracked tokens after reconnect."""
        if not self._subscribed_tokens or not self._ws:
            return
        tokens = list(self._subscribed_tokens)
        for tid in tokens:
            self._local_orderbook.setdefault(tid, {"bids": [], "asks": []})
        msg = json.dumps({"assets_ids": tokens, "type": "MARKET"})
        await self._ws.send(msg)
        logger.info("polymarket_ws_resubscribed", tokens=len(tokens))

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
        *,
        send: bool = True,
    ) -> None:
        """Subscribe to order book updates for a market.

        Args:
            market_id: Condition ID (e.g. "0x123abc…")
            token_map: ``{outcome: token_id}`` mapping.  Required for the CLOB WS
                       to know which token streams to open.  If ``None``, the call
                       is a no-op.
            send: If False, register the tokens locally but do NOT send the WS
                  message yet.  Call ``flush_subscriptions()`` afterwards to send
                  a single batched subscription.  This avoids a rapid-fire race
                  where multiple replace messages within milliseconds cause the
                  server to drop book snapshots.

        If the WebSocket is temporarily disconnected, the subscription is
        recorded locally and will be sent on the next reconnect.
        """
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

        if new_tokens and send:
            await self.flush_subscriptions()
        elif new_tokens:
            logger.debug(
                "clob_ws_queued",
                market_id=market_id,
                tokens=len(new_tokens),
            )

        self._subscriptions.add(("prediction", market_id))

    async def flush_subscriptions(self) -> None:
        """Send the current subscription set to the WS in a single message.

        Safe to call multiple times — only sends if connected.
        """
        if not self._subscribed_tokens or not self._ws or not self._connected:
            return
        all_tokens = list(self._subscribed_tokens)
        msg = json.dumps({"assets_ids": all_tokens, "type": "MARKET"})
        await self._ws.send(msg)
        logger.info(
            "clob_ws_subscribed",
            tokens=len(all_tokens),
        )

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
            except Exception as exc:
                logger.warning("unsubscribe_market_failed", market_id=market_id, error=str(exc))

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
                raise
            except Exception as e:
                logger.warning("keepalive_error", error=str(e))
                return  # exit so _connection_loop reconnects

    async def _receive_loop(self) -> None:
        """Receive and process incoming WebSocket messages."""
        while self._connected and self._ws:
            try:
                message = await self._ws.recv()
                if not isinstance(message, str) or not message.strip():
                    continue  # skip binary or empty frames
                if message.strip() in ("pong", "ping"):
                    continue  # skip keepalive responses
                data = _json_loads(message)
                await self._process_message(data)
            except asyncio.CancelledError:
                raise
            except websockets.exceptions.ConnectionClosed:
                logger.warning("polymarket_ws_connection_closed")
                return  # exit so _connection_loop reconnects
            except (json.JSONDecodeError, ValueError):
                continue  # skip non-JSON messages
            except Exception as e:
                logger.error("receive_loop_error", error=str(e))
                continue

    async def _process_message(self, data: Any) -> None:
        """Process an incoming CLOB WS message.

        Messages are either:
        - list: initial ``book`` snapshots (one per subscribed token)
        - dict: ``book`` refresh, ``price_change``, or ``last_trade_price``
        """
        if isinstance(data, list):
            for item in data:
                event_type = item.get("event_type", "")
                if event_type == "book":
                    self._handle_book_snapshot(item)
        elif isinstance(data, dict):
            event_type = data.get("event_type", "")
            if event_type == "book":
                self._handle_book_snapshot(data)
            elif event_type == "price_change":
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
        self.last_update_ts = time.monotonic()
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


# ---------------------------------------------------------------------------
# User channel — real-time order/trade notifications (authenticated)
# ---------------------------------------------------------------------------

@dataclass
class UserTradeEvent:
    """A fill notification from the Polymarket User WS channel."""

    order_id: str
    market: str  # condition_id
    asset_id: str  # token_id
    side: str
    price: float
    size: float
    status: str  # MATCHED / MINED / CONFIRMED / FAILED
    timestamp: float

    @classmethod
    def from_raw(cls, data: dict[str, Any]) -> "UserTradeEvent":
        maker_orders = data.get("maker_orders") or []
        order_id = data.get("taker_order_id", "")
        if not order_id and maker_orders:
            order_id = maker_orders[0].get("order_id", "")
        return cls(
            order_id=order_id,
            market=str(data.get("market", "")),
            asset_id=str(data.get("asset_id", "")),
            side=str(data.get("side", "")),
            price=float(data.get("price", 0)),
            size=float(data.get("size", 0)),
            status=str(data.get("status", "")),
            timestamp=float(data.get("timestamp", 0)),
        )


class PolymarketUserFeed:
    """Real-time order/trade notifications via Polymarket User WS channel.

    Pushes ``UserTradeEvent`` into an ``asyncio.Queue`` for consumers to drain.
    Requires API credentials (obtained from py-clob-client).

    Auto-reconnects on connection drops with exponential backoff, preserving
    market subscriptions and re-authenticating on each reconnect.
    """

    WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/user"
    PING_INTERVAL = 10.0
    RECONNECT_BASE = 1.0
    RECONNECT_MAX = 30.0

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        api_passphrase: str,
    ) -> None:
        self._api_key = api_key
        self._api_secret = api_secret
        self._api_passphrase = api_passphrase
        self._ws: Optional[ClientConnection] = None
        self._connected = False
        self._subscribed_markets: set[str] = set()
        self._connection_task: Optional[asyncio.Task] = None
        self._shutdown = False
        # Consumers drain this queue for fill events.
        self.fills: asyncio.Queue[UserTradeEvent] = asyncio.Queue()
        # Event signaled on every new fill.
        self.fill_received: asyncio.Event = asyncio.Event()

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def connect(self) -> None:
        if self._connection_task and not self._connection_task.done():
            return
        self._shutdown = False
        self._connection_task = asyncio.create_task(self._connection_loop())

    async def disconnect(self) -> None:
        self._shutdown = True

        if self._connection_task:
            self._connection_task.cancel()
            try:
                await self._connection_task
            except asyncio.CancelledError:
                pass
            self._connection_task = None

        await self._close_ws()
        self._subscribed_markets.clear()

    async def _close_ws(self) -> None:
        self._connected = False
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

    async def _connection_loop(self) -> None:
        """Connect, authenticate, run, and auto-reconnect on failure."""
        backoff = self.RECONNECT_BASE
        while not self._shutdown:
            try:
                self._ws = await websockets.connect(self.WS_URL, proxy=None)
                self._connected = True
                backoff = self.RECONNECT_BASE
                logger.info("user_ws_connected")

                # Re-authenticate and re-subscribe.
                await self._resubscribe_all()

                await asyncio.gather(
                    self._keepalive_loop(),
                    self._receive_loop(),
                )
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("user_ws_disconnected", error=str(exc), reconnect_in=backoff)

            await self._close_ws()

            if self._shutdown:
                break
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, self.RECONNECT_MAX)

    async def _resubscribe_all(self) -> None:
        """Authenticate (required) and re-subscribe all markets after connect/reconnect."""
        if not self._ws:
            return
        # Always send auth — server closes unauthenticated connections quickly.
        market_ids = list(self._subscribed_markets) if self._subscribed_markets else []
        msg: dict[str, Any] = {
            "type": "user",
            "auth": {
                "apiKey": self._api_key,
                "secret": self._api_secret,
                "passphrase": self._api_passphrase,
            },
        }
        if market_ids:
            msg["markets"] = market_ids
        await self._ws.send(json.dumps(msg))
        logger.info("user_ws_authenticated", markets=len(market_ids))

    async def subscribe_markets(self, market_ids: list[str]) -> None:
        """Subscribe to trade/order events for condition IDs.

        If disconnected, subscriptions are recorded and will be sent on reconnect.
        """
        new_ids = [m for m in market_ids if m not in self._subscribed_markets]
        if not new_ids:
            return

        self._subscribed_markets.update(new_ids)

        if not self._ws or not self._connected:
            logger.debug("user_ws_queued_for_reconnect", markets=len(new_ids))
            return

        # Check if this is the first subscribe (needs auth) on this connection.
        # After reconnect, _resubscribe_all already sent auth, so use plain subscribe.
        msg = json.dumps({
            "markets": new_ids,
            "operation": "subscribe",
        })
        await self._ws.send(msg)
        logger.debug("user_ws_subscribed", markets=len(new_ids))

    async def unsubscribe_markets(self, market_ids: list[str]) -> None:
        """Unsubscribe from trade/order events."""
        ids = [m for m in market_ids if m in self._subscribed_markets]
        if not ids:
            return
        self._subscribed_markets -= set(ids)
        if not self._ws or not self._connected:
            return
        try:
            await self._ws.send(json.dumps({
                "markets": ids,
                "operation": "unsubscribe",
            }))
        except Exception as exc:
            logger.warning("user_ws_unsubscribe_failed", error=str(exc))

    async def _keepalive_loop(self) -> None:
        while self._connected and self._ws:
            try:
                await asyncio.sleep(self.PING_INTERVAL)
                if self._ws:
                    await self._ws.ping()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning("user_ws_keepalive_error", error=str(e))
                return  # exit so _connection_loop reconnects

    async def _receive_loop(self) -> None:
        while self._connected and self._ws:
            try:
                message = await self._ws.recv()
                if not isinstance(message, str) or not message.strip():
                    continue
                txt = message.strip()
                if txt in ("PONG", "pong", "ping", "PING"):
                    continue
                data = _json_loads(message)
                self._process_message(data)
            except asyncio.CancelledError:
                raise
            except websockets.exceptions.ConnectionClosed:
                logger.warning("user_ws_connection_closed")
                return  # exit so _connection_loop reconnects
            except (json.JSONDecodeError, ValueError):
                continue
            except Exception as e:
                logger.error("user_ws_receive_error", error=str(e))
                continue

    def _process_message(self, data: Any) -> None:
        if not isinstance(data, dict):
            return
        event_type = data.get("event_type", "")
        if event_type == "trade":
            status = data.get("status", "")
            if status in ("MATCHED", "CONFIRMED"):
                evt = UserTradeEvent.from_raw(data)
                self.fills.put_nowait(evt)
                self.fill_received.set()
                logger.debug(
                    "user_ws_fill",
                    order_id=evt.order_id[:16],
                    market=evt.market[:16],
                    side=evt.side,
                    price=evt.price,
                    size=evt.size,
                    status=evt.status,
                )
