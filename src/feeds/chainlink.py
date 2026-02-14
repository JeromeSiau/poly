"""Polymarket Chainlink price feed via WebSocket.

Connects to wss://ws-live-data.polymarket.com and streams
``crypto_prices_chainlink`` updates for BTC, ETH, SOL, XRP —
the same Chainlink prices used to resolve 15-min crypto markets.

Auto-reconnects with exponential backoff.  Price cache persists
across reconnects so callers always see the last known price.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Optional

import structlog
import websockets
from websockets import ClientConnection

logger = structlog.get_logger()

WS_URL = "wss://ws-live-data.polymarket.com"
PING_INTERVAL = 10.0
RECONNECT_BASE = 1.0
RECONNECT_MAX = 30.0


class ChainlinkFeed:
    """Real-time crypto price feed from Polymarket Chainlink WebSocket.

    Subscribes to the ``crypto_prices_chainlink`` topic and maintains
    the latest price for each symbol (btc/usd, eth/usd, sol/usd, xrp/usd).

    Usage::

        feed = ChainlinkFeed()
        await feed.connect()
        price = feed.get_price("btc/usd")  # -> 69799.00
        await feed.disconnect()
    """

    def __init__(self) -> None:
        self._ws: Optional[ClientConnection] = None
        self._connected: bool = False
        self._prices: dict[str, float] = {}  # "btc/usd" -> 69799.00
        self._last_update_ts: float = 0.0
        self._connection_task: Optional[asyncio.Task] = None
        self._shutdown: bool = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def last_update_ts(self) -> float:
        """Unix timestamp of last price update."""
        return self._last_update_ts

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Start auto-reconnecting WebSocket loop."""
        if self._connection_task and not self._connection_task.done():
            return
        self._shutdown = False
        self._connection_task = asyncio.create_task(self._connection_loop())

    async def disconnect(self) -> None:
        """Close connection and stop reconnecting."""
        self._shutdown = True
        if self._connection_task:
            self._connection_task.cancel()
            try:
                await self._connection_task
            except asyncio.CancelledError:
                pass
            self._connection_task = None
        await self._close_ws()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _close_ws(self) -> None:
        self._connected = False
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

    async def _connection_loop(self) -> None:
        backoff = RECONNECT_BASE
        while not self._shutdown:
            try:
                self._ws = await websockets.connect(WS_URL, proxy=None)
                self._connected = True
                backoff = RECONNECT_BASE
                logger.info("chainlink_ws_connected")

                await self._subscribe()
                await asyncio.gather(
                    self._keepalive_loop(),
                    self._receive_loop(),
                )
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning(
                    "chainlink_ws_disconnected",
                    error=str(exc)[:120],
                    reconnect_in=backoff,
                )

            await self._close_ws()
            if self._shutdown:
                break
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, RECONNECT_MAX)

    async def _subscribe(self) -> None:
        if not self._ws:
            return
        msg = {
            "action": "subscribe",
            "subscriptions": [
                {"topic": "crypto_prices_chainlink", "type": "*", "filters": ""}
            ],
        }
        await self._ws.send(json.dumps(msg))
        logger.info("chainlink_ws_subscribed")

    async def _keepalive_loop(self) -> None:
        while self._connected and self._ws:
            try:
                await asyncio.sleep(PING_INTERVAL)
                if self._ws:
                    await self._ws.ping()
            except asyncio.CancelledError:
                raise
            except Exception:
                return  # exit to trigger reconnect

    async def _receive_loop(self) -> None:
        while self._connected and self._ws:
            try:
                raw = await self._ws.recv()
                if isinstance(raw, str):
                    data = json.loads(raw)
                    self._process_message(data)
            except asyncio.CancelledError:
                raise
            except websockets.exceptions.ConnectionClosed:
                return
            except json.JSONDecodeError:
                continue

    def _process_message(self, data: dict) -> None:
        """Update price cache from a ``crypto_prices_chainlink`` message.

        Expected format::

            {
                "topic": "crypto_prices_chainlink",
                "type": "update",
                "payload": {
                    "symbol": "btc/usd",
                    "value": 69799.00,
                    "timestamp": 1771096165000
                }
            }
        """
        if data.get("topic") != "crypto_prices_chainlink":
            return
        payload = data.get("payload")
        if not payload:
            return

        symbol = payload.get("symbol", "")
        value = payload.get("value")
        if not symbol or value is None:
            return

        try:
            self._prices[symbol.lower()] = float(value)
            self._last_update_ts = time.time()
        except (ValueError, TypeError):
            pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_price(self, symbol: str) -> Optional[float]:
        """Return the latest Chainlink price for *symbol* (e.g. ``"btc/usd"``)."""
        return self._prices.get(symbol.lower())

    def snapshot_price(self, symbol: str) -> Optional[float]:
        """Alias for :meth:`get_price` — semantic sugar for ref-price capture."""
        return self.get_price(symbol)
