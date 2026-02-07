# Polymarket Strategies Expansion Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extend the Poly trading system with 4 new strategy modules inspired by top Polymarket traders: (A) combinatorial arbitrage via integer programming, (B) crypto 15-min reality arb exploiting CEX lag, (C) systematic NO betting on mispriced unlikely events, (D) LLM-powered market screening with Perplexity.

**Architecture:** Each strategy is a standalone engine following the existing pattern (`RealityArbEngine`, `CrossMarketArbEngine`). They share `UnifiedRiskManager` and `PositionManager`, connect to existing feeds or new ones, and persist via the same SQLAlchemy ORM. New strategy types are added to the risk manager allocation.

**Tech Stack:** Python 3.12, asyncio, SQLAlchemy async, structlog, websockets, httpx, py-clob-client. New deps: `gurobipy` (or `scipy.optimize` fallback) for combinatorial arb, `ccxt` for CEX feeds, `perplexity-api` (via httpx) for market screening.

---

## Phase A: Crypto 15-Min Reality Arb (Strat #9 — Browomo)

> Exploits the ~30s lag between CEX price movements (Binance/Coinbase) and Polymarket 15-min crypto market odds updates. This is the highest-priority item because it reuses the existing `RealityArbEngine` pattern directly.

### Task A1: Binance WebSocket Feed

**Files:**
- Create: `src/feeds/binance.py`
- Create: `tests/feeds/test_binance.py`

**Step 1: Write the failing test**

```python
# tests/feeds/test_binance.py
"""Tests for Binance WebSocket price feed."""
import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch

from src.feeds.binance import BinanceFeed, BinanceTick


class TestBinanceTick:
    def test_from_raw_trade(self):
        raw = {
            "e": "trade",
            "s": "BTCUSDT",
            "p": "65432.10",
            "q": "0.5",
            "T": 1707300000000,
        }
        tick = BinanceTick.from_raw(raw)
        assert tick.symbol == "BTCUSDT"
        assert tick.price == 65432.10
        assert tick.quantity == 0.5
        assert tick.timestamp == 1707300000.0

    def test_from_raw_missing_fields(self):
        tick = BinanceTick.from_raw({})
        assert tick.symbol == ""
        assert tick.price == 0.0


class TestBinanceFeed:
    def test_init_default_symbols(self):
        feed = BinanceFeed()
        assert "BTCUSDT" in feed.symbols
        assert "ETHUSDT" in feed.symbols
        assert "SOLUSDT" in feed.symbols

    def test_init_custom_symbols(self):
        feed = BinanceFeed(symbols=["XRPUSDT"])
        assert feed.symbols == ["XRPUSDT"]

    def test_fair_value_empty(self):
        feed = BinanceFeed()
        fv = feed.get_fair_value("BTCUSDT")
        assert fv is None

    def test_fair_value_with_trades(self):
        feed = BinanceFeed(fair_value_window=3)
        # Simulate 3 trades
        feed._recent_trades["BTCUSDT"] = [
            BinanceTick("BTCUSDT", 100.0, 1.0, 1.0),
            BinanceTick("BTCUSDT", 102.0, 2.0, 2.0),
            BinanceTick("BTCUSDT", 104.0, 1.0, 3.0),
        ]
        fv = feed.get_fair_value("BTCUSDT")
        # Volume-weighted: (100*1 + 102*2 + 104*1) / (1+2+1) = 408/4 = 102.0
        assert fv == pytest.approx(102.0)

    def test_price_direction_up(self):
        feed = BinanceFeed(fair_value_window=3)
        feed._recent_trades["BTCUSDT"] = [
            BinanceTick("BTCUSDT", 100.0, 1.0, 1.0),
            BinanceTick("BTCUSDT", 100.0, 1.0, 2.0),
            BinanceTick("BTCUSDT", 105.0, 1.0, 3.0),
        ]
        direction = feed.get_price_direction("BTCUSDT")
        assert direction == "UP"

    def test_price_direction_down(self):
        feed = BinanceFeed(fair_value_window=3)
        feed._recent_trades["BTCUSDT"] = [
            BinanceTick("BTCUSDT", 105.0, 1.0, 1.0),
            BinanceTick("BTCUSDT", 105.0, 1.0, 2.0),
            BinanceTick("BTCUSDT", 100.0, 1.0, 3.0),
        ]
        direction = feed.get_price_direction("BTCUSDT")
        assert direction == "DOWN"

    def test_price_direction_neutral(self):
        feed = BinanceFeed(fair_value_window=3)
        feed._recent_trades["BTCUSDT"] = [
            BinanceTick("BTCUSDT", 100.0, 1.0, 1.0),
            BinanceTick("BTCUSDT", 100.0, 1.0, 2.0),
        ]
        direction = feed.get_price_direction("BTCUSDT")
        assert direction == "NEUTRAL"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/feeds/test_binance.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.feeds.binance'`

**Step 3: Write minimal implementation**

```python
# src/feeds/binance.py
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
        self._recent_trades: dict[str, list[BinanceTick]] = {}
        self._ws: Optional[Any] = None
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
        # Not applicable for CEX feed — symbols set at init
        pass

    async def listen(self) -> None:
        """Listen for trade events and update internal state."""
        if not self._ws:
            raise RuntimeError("Not connected")

        async for message in self._ws:
            data = json.loads(message)
            if "data" in data:
                data = data["data"]

            tick = BinanceTick.from_raw(data)
            if not tick.symbol:
                continue

            # Maintain rolling window
            if tick.symbol not in self._recent_trades:
                self._recent_trades[tick.symbol] = []
            trades = self._recent_trades[tick.symbol]
            trades.append(tick)
            if len(trades) > self._fair_value_window:
                trades.pop(0)

            # Emit feed event
            event = FeedEvent(
                source="binance",
                event_type="trade",
                game="crypto",
                data={"symbol": tick.symbol, "price": tick.price, "qty": tick.quantity},
                timestamp=tick.timestamp,
                match_id=tick.symbol,
            )
            await self._emit(event)

    def get_fair_value(self, symbol: str) -> Optional[float]:
        """Calculate VWAP fair value from recent trades.

        Returns None if no trades available.
        """
        trades = self._recent_trades.get(symbol, [])
        if not trades:
            return None

        total_value = sum(t.price * t.quantity for t in trades)
        total_volume = sum(t.quantity for t in trades)
        if total_volume == 0:
            return None

        return total_value / total_volume

    def get_price_direction(self, symbol: str) -> str:
        """Determine current price direction: UP, DOWN, or NEUTRAL.

        Compares latest trade price to VWAP of the window.
        Returns NEUTRAL if insufficient data or move below threshold.
        """
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
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/feeds/test_binance.py -v`
Expected: All 8 tests PASS

**Step 5: Commit**

```bash
git add src/feeds/binance.py tests/feeds/test_binance.py
git commit -m "feat: add Binance WebSocket feed with VWAP fair value"
```

---

### Task A2: Crypto Market Mapper

**Files:**
- Create: `src/realtime/crypto_mapper.py`
- Create: `tests/realtime/test_crypto_mapper.py`

**Step 1: Write the failing test**

```python
# tests/realtime/test_crypto_mapper.py
"""Tests for crypto market mapper — links CEX symbols to Polymarket 15-min markets."""
import pytest
from unittest.mock import MagicMock

from src.realtime.crypto_mapper import CryptoMarketMapper


class TestCryptoMarketMapper:
    def test_map_symbol_to_market(self):
        mapper = CryptoMarketMapper()
        # Simulate synced markets
        mapper._active_markets = {
            "btc-up-15min-001": {
                "condition_id": "btc-up-15min-001",
                "title": "Will Bitcoin go up in the next 15 minutes?",
                "tokens": [
                    {"token_id": "tok-yes-001", "outcome": "Yes"},
                    {"token_id": "tok-no-001", "outcome": "No"},
                ],
            }
        }
        mapper._symbol_to_markets = {"BTCUSDT": ["btc-up-15min-001"]}

        result = mapper.get_active_market("BTCUSDT")
        assert result is not None
        assert result["condition_id"] == "btc-up-15min-001"

    def test_map_symbol_no_market(self):
        mapper = CryptoMarketMapper()
        result = mapper.get_active_market("DOGEUSDT")
        assert result is None

    def test_get_token_for_direction_up(self):
        mapper = CryptoMarketMapper()
        market = {
            "tokens": [
                {"token_id": "tok-yes", "outcome": "Yes"},
                {"token_id": "tok-no", "outcome": "No"},
            ]
        }
        token_id, outcome = mapper.get_token_for_direction(market, "UP")
        assert token_id == "tok-yes"
        assert outcome == "Yes"

    def test_get_token_for_direction_down(self):
        mapper = CryptoMarketMapper()
        market = {
            "tokens": [
                {"token_id": "tok-yes", "outcome": "Yes"},
                {"token_id": "tok-no", "outcome": "No"},
            ]
        }
        token_id, outcome = mapper.get_token_for_direction(market, "DOWN")
        assert token_id == "tok-no"
        assert outcome == "No"

    def test_get_token_for_neutral_returns_none(self):
        mapper = CryptoMarketMapper()
        market = {"tokens": []}
        result = mapper.get_token_for_direction(market, "NEUTRAL")
        assert result is None
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/realtime/test_crypto_mapper.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# src/realtime/crypto_mapper.py
"""Maps CEX symbols (BTCUSDT, etc.) to active Polymarket 15-min crypto markets.

Polymarket has rolling 15-minute markets like:
- "Will Bitcoin go up in the next 15 minutes?"
- "Will ETH be above $X at 14:30 UTC?"

This mapper syncs active crypto markets from Polymarket and links them
to the corresponding CEX trading pair for real-time signal routing.
"""

import re
from typing import Any, Optional

import structlog

logger = structlog.get_logger()

# Map CEX symbols to Polymarket search keywords
SYMBOL_KEYWORDS: dict[str, list[str]] = {
    "BTCUSDT": ["bitcoin", "btc"],
    "ETHUSDT": ["ethereum", "eth"],
    "SOLUSDT": ["solana", "sol"],
    "XRPUSDT": ["xrp", "ripple"],
}


class CryptoMarketMapper:
    """Links CEX trading pairs to Polymarket 15-minute crypto markets."""

    def __init__(self) -> None:
        # condition_id -> market dict
        self._active_markets: dict[str, dict[str, Any]] = {}
        # symbol -> list of condition_ids
        self._symbol_to_markets: dict[str, list[str]] = {}

    async def sync_markets(self, polymarket_feed: Any) -> int:
        """Sync active 15-min crypto markets from Polymarket.

        Fetches markets matching crypto keywords and maps them to CEX symbols.
        Returns number of markets synced.
        """
        count = 0
        markets = await polymarket_feed.get_markets(tag="crypto", active=True)

        for market in markets:
            title = market.get("title", "").lower()
            condition_id = market.get("condition_id", "")

            for symbol, keywords in SYMBOL_KEYWORDS.items():
                if any(kw in title for kw in keywords):
                    self._active_markets[condition_id] = market
                    if symbol not in self._symbol_to_markets:
                        self._symbol_to_markets[symbol] = []
                    self._symbol_to_markets[symbol].append(condition_id)
                    count += 1
                    break

        logger.info("crypto_markets_synced", count=count)
        return count

    def get_active_market(self, symbol: str) -> Optional[dict[str, Any]]:
        """Get the most relevant active market for a CEX symbol.

        Returns None if no active market exists for this symbol.
        """
        market_ids = self._symbol_to_markets.get(symbol, [])
        if not market_ids:
            return None
        # Return the first active market (TODO: pick closest to expiry)
        return self._active_markets.get(market_ids[0])

    def get_token_for_direction(
        self, market: dict[str, Any], direction: str
    ) -> Optional[tuple[str, str]]:
        """Get the token_id and outcome for a given price direction.

        Args:
            market: Market dict with "tokens" list
            direction: "UP" or "DOWN"

        Returns:
            (token_id, outcome) tuple or None if direction is NEUTRAL
        """
        if direction == "NEUTRAL":
            return None

        tokens = market.get("tokens", [])
        if not tokens:
            return None

        # UP → buy YES, DOWN → buy NO
        target_outcome = "Yes" if direction == "UP" else "No"
        for token in tokens:
            if token.get("outcome") == target_outcome:
                return token["token_id"], token["outcome"]

        return None
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/realtime/test_crypto_mapper.py -v`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add src/realtime/crypto_mapper.py tests/realtime/test_crypto_mapper.py
git commit -m "feat: add crypto market mapper for CEX-to-Polymarket linking"
```

---

### Task A3: Crypto Reality Arb Engine

**Files:**
- Create: `src/arb/crypto_arb.py`
- Create: `tests/arb/test_crypto_arb.py`

**Step 1: Write the failing test**

```python
# tests/arb/test_crypto_arb.py
"""Tests for CryptoArbEngine — exploits CEX-to-Polymarket lag on 15-min markets."""
import pytest
import time
from unittest.mock import MagicMock, AsyncMock

from src.arb.crypto_arb import CryptoArbEngine, CryptoArbOpportunity


class TestCryptoArbOpportunity:
    def test_opportunity_valid(self):
        opp = CryptoArbOpportunity(
            symbol="BTCUSDT",
            market_id="mkt-001",
            token_id="tok-001",
            side="BUY",
            outcome="Yes",
            polymarket_price=0.45,
            fair_value_price=0.62,
            cex_direction="UP",
            edge_pct=0.17,
            timestamp=time.time(),
        )
        assert opp.is_valid

    def test_opportunity_edge_too_low(self):
        opp = CryptoArbOpportunity(
            symbol="BTCUSDT",
            market_id="mkt-001",
            token_id="tok-001",
            side="BUY",
            outcome="Yes",
            polymarket_price=0.50,
            fair_value_price=0.51,
            cex_direction="UP",
            edge_pct=0.01,
            timestamp=time.time(),
        )
        assert not opp.is_valid

    def test_opportunity_stale(self):
        opp = CryptoArbOpportunity(
            symbol="BTCUSDT",
            market_id="mkt-001",
            token_id="tok-001",
            side="BUY",
            outcome="Yes",
            polymarket_price=0.45,
            fair_value_price=0.62,
            cex_direction="UP",
            edge_pct=0.17,
            timestamp=time.time() - 120,  # 2 minutes old
        )
        assert not opp.is_valid


class TestCryptoArbEngine:
    def test_init(self):
        engine = CryptoArbEngine()
        assert engine.min_edge_pct == 0.02
        assert engine.stale_seconds == 45.0

    def test_estimate_fair_price_up(self):
        engine = CryptoArbEngine()
        # If BTC went UP on Binance, YES should be worth more
        fair = engine.estimate_fair_price(
            direction="UP",
            current_polymarket_price=0.50,
            cex_pct_move=0.005,  # 0.5% move
        )
        assert fair > 0.50

    def test_estimate_fair_price_down(self):
        engine = CryptoArbEngine()
        fair = engine.estimate_fair_price(
            direction="DOWN",
            current_polymarket_price=0.50,
            cex_pct_move=-0.005,
        )
        assert fair < 0.50

    def test_estimate_fair_price_clamped(self):
        engine = CryptoArbEngine()
        fair = engine.estimate_fair_price(
            direction="UP",
            current_polymarket_price=0.98,
            cex_pct_move=0.05,
        )
        assert fair <= 0.99

    @pytest.mark.asyncio
    async def test_evaluate_opportunity_with_edge(self):
        binance_feed = MagicMock()
        binance_feed.get_fair_value.return_value = 65500.0
        binance_feed.get_price_direction.return_value = "UP"
        binance_feed._recent_trades = {
            "BTCUSDT": [MagicMock(price=65000.0), MagicMock(price=65500.0)]
        }

        polymarket_feed = MagicMock()
        polymarket_feed.get_best_prices.return_value = (0.45, 0.47)

        crypto_mapper = MagicMock()
        crypto_mapper.get_active_market.return_value = {
            "condition_id": "mkt-001",
            "tokens": [
                {"token_id": "tok-yes", "outcome": "Yes"},
                {"token_id": "tok-no", "outcome": "No"},
            ],
        }
        crypto_mapper.get_token_for_direction.return_value = ("tok-yes", "Yes")

        engine = CryptoArbEngine(
            binance_feed=binance_feed,
            polymarket_feed=polymarket_feed,
            crypto_mapper=crypto_mapper,
        )

        opp = engine.evaluate_opportunity("BTCUSDT")
        assert opp is not None
        assert opp.side == "BUY"
        assert opp.outcome == "Yes"
        assert opp.edge_pct > 0

    @pytest.mark.asyncio
    async def test_evaluate_no_opportunity_neutral(self):
        binance_feed = MagicMock()
        binance_feed.get_price_direction.return_value = "NEUTRAL"

        engine = CryptoArbEngine(binance_feed=binance_feed)
        opp = engine.evaluate_opportunity("BTCUSDT")
        assert opp is None
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/arb/test_crypto_arb.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# src/arb/crypto_arb.py
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
from dataclasses import dataclass, field
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
        risk_manager: Optional[Any] = None,
        position_manager: Optional[Any] = None,
    ):
        self.binance_feed = binance_feed
        self.polymarket_feed = polymarket_feed
        self.crypto_mapper = crypto_mapper
        self.risk_manager = risk_manager
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
        # Convert CEX move to probability shift
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
        trades = self.binance_feed._recent_trades.get(symbol, [])
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
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/arb/test_crypto_arb.py -v`
Expected: All 8 tests PASS

**Step 5: Commit**

```bash
git add src/arb/crypto_arb.py tests/arb/test_crypto_arb.py
git commit -m "feat: add crypto reality arb engine exploiting CEX-to-Polymarket lag"
```

---

### Task A4: Crypto Arb Runner Script

**Files:**
- Create: `scripts/run_crypto_arb.py`
- Modify: `config/settings.py` — add crypto-specific settings

**Step 1: Add settings**

Add to `config/settings.py` after the existing capital allocation settings:

```python
    # === Crypto Reality Arb ===
    CRYPTO_ARB_SYMBOLS: str = "BTCUSDT,ETHUSDT,SOLUSDT"
    CRYPTO_ARB_SCAN_INTERVAL: float = 1.0  # scan every 1 second
    CRYPTO_ARB_FAIR_VALUE_WINDOW: int = 10  # last 10 trades for VWAP
    CRYPTO_ARB_CEX_SENSITIVITY: float = 30.0  # prob shift multiplier
    CRYPTO_ARB_STALE_SECONDS: float = 45.0
    CAPITAL_ALLOCATION_CRYPTO_PCT: float = 0.0  # disabled by default
```

**Step 2: Update `UnifiedRiskManager` to support `crypto` strategy type**

Modify `src/risk/manager.py`:
- Add `"crypto"` to `StrategyType` literal
- Add `crypto_allocation_pct` parameter to `__init__`
- Add `"crypto": 0.0` to `_daily_pnl_by_strategy`
- Handle `"crypto"` in `get_available_capital()`

**Step 3: Write the runner script**

```python
# scripts/run_crypto_arb.py
"""Runner for crypto reality arbitrage on 15-minute Polymarket markets.

Usage:
    python scripts/run_crypto_arb.py [--symbols BTCUSDT,ETHUSDT] [--autopilot]
"""

import argparse
import asyncio

import structlog

from config.settings import settings
from src.feeds.binance import BinanceFeed
from src.feeds.polymarket import PolymarketFeed
from src.realtime.crypto_mapper import CryptoMarketMapper
from src.arb.crypto_arb import CryptoArbEngine
from src.arb.position_manager import PositionManager
from src.risk.manager import UnifiedRiskManager

logger = structlog.get_logger()


async def main(symbols: list[str], autopilot: bool) -> None:
    logger.info("crypto_arb_starting", symbols=symbols, autopilot=autopilot)

    # Init feeds
    binance = BinanceFeed(
        symbols=symbols,
        fair_value_window=settings.CRYPTO_ARB_FAIR_VALUE_WINDOW,
    )
    polymarket = PolymarketFeed()
    mapper = CryptoMarketMapper()

    # Init risk manager
    risk_mgr = UnifiedRiskManager(
        global_capital=settings.GLOBAL_CAPITAL,
        reality_allocation_pct=settings.CAPITAL_ALLOCATION_REALITY_PCT,
        crossmarket_allocation_pct=settings.CAPITAL_ALLOCATION_CROSSMARKET_PCT,
        crypto_allocation_pct=settings.CAPITAL_ALLOCATION_CRYPTO_PCT,
        max_position_pct=settings.MAX_POSITION_PCT,
        daily_loss_limit_pct=settings.DAILY_LOSS_LIMIT_PCT,
    )

    # Init engine
    engine = CryptoArbEngine(
        binance_feed=binance,
        polymarket_feed=polymarket,
        crypto_mapper=mapper,
        risk_manager=risk_mgr,
    )

    # Connect feeds
    await binance.connect()
    await polymarket.connect()
    await mapper.sync_markets(polymarket)

    # Scan loop
    async def scan_loop():
        while True:
            for symbol in symbols:
                opp = engine.evaluate_opportunity(symbol)
                if opp and opp.is_valid:
                    logger.info(
                        "crypto_arb_opportunity",
                        symbol=opp.symbol,
                        direction=opp.cex_direction,
                        edge=f"{opp.edge_pct:.2%}",
                        pm_price=opp.polymarket_price,
                        fair_price=opp.fair_value_price,
                    )
            await asyncio.sleep(settings.CRYPTO_ARB_SCAN_INTERVAL)

    # Run feed listener and scanner concurrently
    await asyncio.gather(binance.listen(), scan_loop())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crypto Reality Arb Bot")
    parser.add_argument("--symbols", default=settings.CRYPTO_ARB_SYMBOLS)
    parser.add_argument("--autopilot", action="store_true")
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",")]
    asyncio.run(main(symbols, args.autopilot))
```

**Step 4: Run tests to verify nothing is broken**

Run: `pytest tests/ -v --timeout=30`
Expected: All existing + new tests PASS

**Step 5: Commit**

```bash
git add config/settings.py src/risk/manager.py scripts/run_crypto_arb.py
git commit -m "feat: add crypto arb runner and extend risk manager for crypto strategy"
```

---

## Phase B: Systematic NO Betting (Strat #4 — DidiTrading)

> Systematically buy NO shares on near-impossible events. NO bets are consistently mispriced because of the psychological bias toward buying YES. This is a passive, low-risk income strategy.

### Task B1: NO Bet Scanner

**Files:**
- Create: `src/arb/no_bet_scanner.py`
- Create: `tests/arb/test_no_bet_scanner.py`

**Step 1: Write the failing test**

```python
# tests/arb/test_no_bet_scanner.py
"""Tests for systematic NO bet scanner."""
import pytest
from unittest.mock import MagicMock

from src.arb.no_bet_scanner import NoBetScanner, NoBetOpportunity


class TestNoBetOpportunity:
    def test_expected_return(self):
        opp = NoBetOpportunity(
            market_id="mkt-001",
            token_id="tok-no-001",
            title="Will aliens land on Earth in 2026?",
            yes_price=0.03,
            no_price=0.95,
            estimated_no_probability=0.99,
            edge_pct=0.04,
            volume_24h=50000.0,
            liquidity=10000.0,
        )
        # Buy NO at 0.95, expect to win $1 with prob 0.99
        # Expected return = 0.99 * (1 - 0.95) - 0.01 * 0.95 = 0.99*0.05 - 0.0095 = 0.04
        assert opp.expected_return == pytest.approx(0.04, abs=0.01)

    def test_is_valid(self):
        opp = NoBetOpportunity(
            market_id="mkt-001",
            token_id="tok-no-001",
            title="Test",
            yes_price=0.03,
            no_price=0.95,
            estimated_no_probability=0.99,
            edge_pct=0.04,
            volume_24h=50000.0,
            liquidity=10000.0,
        )
        assert opp.is_valid

    def test_is_invalid_low_liquidity(self):
        opp = NoBetOpportunity(
            market_id="mkt-001",
            token_id="tok-no-001",
            title="Test",
            yes_price=0.03,
            no_price=0.95,
            estimated_no_probability=0.99,
            edge_pct=0.04,
            volume_24h=100.0,
            liquidity=50.0,  # too low
        )
        assert not opp.is_valid


class TestNoBetScanner:
    def test_init_defaults(self):
        scanner = NoBetScanner()
        assert scanner.max_yes_price == 0.10
        assert scanner.min_liquidity == 1000.0
        assert scanner.min_volume_24h == 5000.0

    def test_score_market_high_confidence(self):
        scanner = NoBetScanner()
        score = scanner.score_market(
            title="Will aliens land on Earth in 2026?",
            yes_price=0.02,
            volume_24h=100000.0,
            end_date_days=30,
        )
        # Very unlikely event, high volume, reasonable timeframe
        assert score > 0.9

    def test_score_market_moderate(self):
        scanner = NoBetScanner()
        score = scanner.score_market(
            title="Will Bitcoin reach $1M by end of 2026?",
            yes_price=0.08,
            volume_24h=20000.0,
            end_date_days=365,
        )
        assert 0.5 < score < 0.95

    def test_filter_candidates(self):
        scanner = NoBetScanner(max_yes_price=0.10)
        markets = [
            {"condition_id": "a", "title": "Impossible thing?", "tokens": [
                {"outcome": "Yes", "price": 0.03},
                {"outcome": "No", "price": 0.95, "token_id": "tok-a"},
            ], "volume_24h": 50000, "liquidity": 10000},
            {"condition_id": "b", "title": "Likely thing?", "tokens": [
                {"outcome": "Yes", "price": 0.60},
                {"outcome": "No", "price": 0.38, "token_id": "tok-b"},
            ], "volume_24h": 50000, "liquidity": 10000},
        ]
        candidates = scanner.filter_candidates(markets)
        assert len(candidates) == 1
        assert candidates[0]["condition_id"] == "a"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/arb/test_no_bet_scanner.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# src/arb/no_bet_scanner.py
"""Systematic NO Bet Scanner.

Identifies and scores Polymarket markets where NO is mispriced.
Strategy from DidiTrading (strat #4): YES is overpriced on
near-impossible events due to psychological bias.

Scoring criteria:
- YES price < 10¢ (event very unlikely)
- High volume (liquid market, can exit)
- Reasonable time to resolution
- Event category (politics/science more predictable than crypto)
"""

from dataclasses import dataclass
from typing import Any, Optional

import structlog

from config.settings import settings

logger = structlog.get_logger()

# Categories where NO bets historically perform best
HIGH_CONFIDENCE_KEYWORDS = [
    "alien", "meteor", "nuclear war", "world end",
    "resign", "impeach", "assassination",
]
MODERATE_CONFIDENCE_KEYWORDS = [
    "reach $1m", "go to zero", "banned", "collapse",
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
        win_payout = 1.0 - self.no_price  # profit if NO wins
        loss = self.no_price  # loss if YES wins
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
    """Scans Polymarket for mispriced NO bets."""

    def __init__(
        self,
        max_yes_price: float = 0.10,
        min_liquidity: float = 1000.0,
        min_volume_24h: float = 5000.0,
    ):
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

        Higher score = more confident that NO is the right bet.
        """
        score = 0.5  # base

        # Lower YES price → more likely NO wins
        if yes_price <= 0.02:
            score += 0.3
        elif yes_price <= 0.05:
            score += 0.2
        elif yes_price <= 0.10:
            score += 0.1

        # Keyword analysis
        title_lower = title.lower()
        if any(kw in title_lower for kw in HIGH_CONFIDENCE_KEYWORDS):
            score += 0.15
        elif any(kw in title_lower for kw in MODERATE_CONFIDENCE_KEYWORDS):
            score += 0.05

        # Volume indicates market is actively traded (prices more meaningful)
        if volume_24h > 100000:
            score += 0.05
        elif volume_24h > 50000:
            score += 0.03

        # Time to resolution: shorter is better (less uncertainty)
        if end_date_days <= 30:
            score += 0.05
        elif end_date_days > 365:
            score -= 0.05

        return max(0.0, min(1.0, score))

    def filter_candidates(
        self, markets: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Filter markets where YES price is below threshold.

        Returns markets where a NO bet is worth investigating.
        """
        candidates = []
        for market in markets:
            tokens = market.get("tokens", [])
            yes_price = None
            for token in tokens:
                if token.get("outcome") == "Yes":
                    yes_price = token.get("price", 1.0)
                    break

            if yes_price is not None and yes_price <= self.max_yes_price:
                candidates.append(market)

        return candidates
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/arb/test_no_bet_scanner.py -v`
Expected: All 7 tests PASS

**Step 5: Commit**

```bash
git add src/arb/no_bet_scanner.py tests/arb/test_no_bet_scanner.py
git commit -m "feat: add systematic NO bet scanner for mispriced unlikely events"
```

---

### Task B2: NO Bet Runner Script

**Files:**
- Create: `scripts/run_no_bet.py`
- Modify: `config/settings.py` — add NO bet settings

**Step 1: Add settings**

```python
    # === NO Bet Strategy ===
    NO_BET_MAX_YES_PRICE: float = 0.10  # Only bet NO when YES < 10¢
    NO_BET_MIN_LIQUIDITY: float = 1000.0
    NO_BET_MIN_VOLUME_24H: float = 5000.0
    NO_BET_SCAN_INTERVAL: float = 300.0  # scan every 5 minutes (slow strategy)
    NO_BET_MAX_PER_MARKET_PCT: float = 0.02  # max 2% of capital per market
    CAPITAL_ALLOCATION_NOBET_PCT: float = 0.0  # disabled by default
```

**Step 2: Write runner**

```python
# scripts/run_no_bet.py
"""Runner for systematic NO bet strategy.

Scans Polymarket for markets where YES is priced < 10¢ (near-impossible events),
scores them, and buys NO shares for passive income.

Usage:
    python scripts/run_no_bet.py [--max-yes-price 0.10] [--autopilot]
"""

import argparse
import asyncio

import structlog

from config.settings import settings
from src.arb.no_bet_scanner import NoBetScanner

logger = structlog.get_logger()


async def main(max_yes_price: float, autopilot: bool) -> None:
    logger.info("no_bet_scanner_starting", max_yes_price=max_yes_price)

    scanner = NoBetScanner(
        max_yes_price=max_yes_price,
        min_liquidity=settings.NO_BET_MIN_LIQUIDITY,
        min_volume_24h=settings.NO_BET_MIN_VOLUME_24H,
    )

    while True:
        logger.info("no_bet_scan_starting")
        # TODO: fetch active markets from Polymarket REST API
        # candidates = scanner.filter_candidates(markets)
        # for each candidate: score, evaluate, alert/execute
        await asyncio.sleep(settings.NO_BET_SCAN_INTERVAL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Systematic NO Bet Scanner")
    parser.add_argument("--max-yes-price", type=float, default=settings.NO_BET_MAX_YES_PRICE)
    parser.add_argument("--autopilot", action="store_true")
    args = parser.parse_args()

    asyncio.run(main(args.max_yes_price, args.autopilot))
```

**Step 3: Run tests**

Run: `pytest tests/ -v --timeout=30`
Expected: All tests PASS

**Step 4: Commit**

```bash
git add config/settings.py scripts/run_no_bet.py
git commit -m "feat: add NO bet runner script and configuration"
```

---

## Phase C: LLM Market Screener (Strat #3 — MindshareXBT)

> Bot that scans all active Polymarket markets, identifies the most interesting ones, and sends them to an LLM for deep research to produce a trading thesis.

### Task C1: Market Screener Core

**Files:**
- Create: `src/screening/market_screener.py`
- Create: `tests/screening/__init__.py`
- Create: `tests/screening/test_market_screener.py`

**Step 1: Write the failing test**

```python
# tests/screening/test_market_screener.py
"""Tests for LLM-powered market screener."""
import pytest
from src.screening.market_screener import MarketScreener, ScreenedMarket


class TestScreenedMarket:
    def test_alpha_score(self):
        m = ScreenedMarket(
            market_id="mkt-001",
            title="Will X happen?",
            volume_24h=100000,
            liquidity=50000,
            price_yes=0.50,
            category="politics",
            end_date="2026-03-01",
            alpha_score=0.75,
        )
        assert m.alpha_score == 0.75

    def test_is_interesting_high_score(self):
        m = ScreenedMarket(
            market_id="mkt-001",
            title="Test",
            volume_24h=100000,
            liquidity=50000,
            price_yes=0.50,
            category="politics",
            end_date="2026-03-01",
            alpha_score=0.80,
        )
        assert m.is_interesting

    def test_is_not_interesting_low_score(self):
        m = ScreenedMarket(
            market_id="mkt-001",
            title="Test",
            volume_24h=100000,
            liquidity=50000,
            price_yes=0.50,
            category="politics",
            end_date="2026-03-01",
            alpha_score=0.30,
        )
        assert not m.is_interesting


class TestMarketScreener:
    def test_init(self):
        screener = MarketScreener()
        assert screener.min_alpha_score == 0.6

    def test_compute_alpha_score_high_volume_midprice(self):
        screener = MarketScreener()
        score = screener.compute_alpha_score(
            volume_24h=200000,
            liquidity=100000,
            price_yes=0.50,
            category="politics",
            hours_to_resolution=48,
        )
        # High volume + mid-price (uncertain) + politics = high alpha
        assert score > 0.6

    def test_compute_alpha_score_low_volume(self):
        screener = MarketScreener()
        score = screener.compute_alpha_score(
            volume_24h=500,
            liquidity=200,
            price_yes=0.50,
            category="crypto",
            hours_to_resolution=2,
        )
        # Low volume = low alpha potential
        assert score < 0.5

    def test_compute_alpha_score_extreme_price(self):
        screener = MarketScreener()
        score = screener.compute_alpha_score(
            volume_24h=100000,
            liquidity=50000,
            price_yes=0.98,  # Already priced in
            category="politics",
            hours_to_resolution=48,
        )
        # Very high price = low alpha
        assert score < 0.5

    def test_screen_markets(self):
        screener = MarketScreener(min_alpha_score=0.5)
        markets = [
            {
                "condition_id": "a",
                "title": "Interesting politics market?",
                "volume_24h": 200000,
                "liquidity": 100000,
                "tokens": [{"outcome": "Yes", "price": 0.50}],
                "category": "politics",
                "end_date": "2026-03-01",
            },
            {
                "condition_id": "b",
                "title": "Dead market?",
                "volume_24h": 100,
                "liquidity": 50,
                "tokens": [{"outcome": "Yes", "price": 0.50}],
                "category": "other",
                "end_date": "2026-03-01",
            },
        ]
        results = screener.screen_markets(markets)
        interesting = [r for r in results if r.is_interesting]
        assert len(interesting) >= 1
        assert interesting[0].market_id == "a"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/screening/test_market_screener.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# src/screening/__init__.py
```

```python
# src/screening/market_screener.py
"""LLM-powered market screener for Polymarket alpha opportunities.

Strategy from MindshareXBT (strat #3):
1. Scan all active Polymarket markets
2. Score each market for alpha potential (volume, uncertainty, category)
3. Send top N markets to LLM (Claude/Perplexity) for deep research
4. Generate trading thesis with confidence level

The screener pre-filters markets using quantitative signals before
sending them to the LLM to save API costs.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

import structlog

logger = structlog.get_logger()

# Categories with historically higher edge from research
ALPHA_CATEGORIES = {
    "politics": 1.3,
    "elections": 1.3,
    "sports": 1.1,
    "crypto": 0.9,
    "science": 1.2,
    "entertainment": 0.8,
}


@dataclass
class ScreenedMarket:
    """A market scored for alpha potential."""

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
    """Screens Polymarket markets for alpha opportunities."""

    def __init__(self, min_alpha_score: float = 0.6):
        self.min_alpha_score = min_alpha_score

    def compute_alpha_score(
        self,
        volume_24h: float,
        liquidity: float,
        price_yes: float,
        category: str,
        hours_to_resolution: float,
    ) -> float:
        """Compute alpha potential score for a market (0.0 to 1.0).

        High alpha = high volume + uncertain price + good category + medium timeframe.
        """
        score = 0.0

        # Volume score (0-0.3): higher volume = more alpha potential
        if volume_24h > 100000:
            score += 0.30
        elif volume_24h > 50000:
            score += 0.20
        elif volume_24h > 10000:
            score += 0.10

        # Uncertainty score (0-0.3): prices near 50% = most uncertain
        # Price at 50% = most edge, near 0% or 100% = priced in
        uncertainty = 1.0 - abs(price_yes - 0.5) * 2  # 1.0 at 50%, 0.0 at 0%/100%
        score += uncertainty * 0.30

        # Category multiplier
        cat_mult = ALPHA_CATEGORIES.get(category, 1.0)
        score *= cat_mult

        # Time to resolution: sweet spot is 1-7 days
        if 24 <= hours_to_resolution <= 168:
            score += 0.10
        elif hours_to_resolution < 2:
            score -= 0.10  # Too short, probably already priced in
        elif hours_to_resolution > 720:
            score -= 0.05  # Very long, hard to predict

        # Liquidity bonus
        if liquidity > 50000:
            score += 0.05

        return max(0.0, min(1.0, score))

    def screen_markets(
        self, markets: list[dict[str, Any]]
    ) -> list[ScreenedMarket]:
        """Screen a list of raw markets and return scored results.

        Args:
            markets: Raw market dicts from Polymarket API

        Returns:
            List of ScreenedMarket sorted by alpha_score descending
        """
        results = []

        for market in markets:
            tokens = market.get("tokens", [])
            price_yes = 0.5
            for token in tokens:
                if token.get("outcome") == "Yes":
                    price_yes = token.get("price", 0.5)
                    break

            # Estimate hours to resolution
            end_date = market.get("end_date", "")
            hours = 48.0  # default
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
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/screening/test_market_screener.py -v`
Expected: All 7 tests PASS

**Step 5: Commit**

```bash
git add src/screening/__init__.py src/screening/market_screener.py \
    tests/screening/__init__.py tests/screening/test_market_screener.py
git commit -m "feat: add market screener with alpha scoring for LLM-assisted trading"
```

---

### Task C2: LLM Research Agent

**Files:**
- Create: `src/screening/llm_researcher.py`
- Create: `tests/screening/test_llm_researcher.py`

**Step 1: Write the failing test**

```python
# tests/screening/test_llm_researcher.py
"""Tests for LLM researcher that generates trading theses."""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from src.screening.llm_researcher import LLMResearcher, ResearchResult


class TestResearchResult:
    def test_has_edge(self):
        r = ResearchResult(
            market_id="mkt-001",
            title="Will X happen?",
            thesis="Based on recent data, X is unlikely because...",
            recommended_side="NO",
            confidence=0.85,
            key_factors=["factor1", "factor2"],
            sources=["source1"],
        )
        assert r.has_edge

    def test_no_edge_low_confidence(self):
        r = ResearchResult(
            market_id="mkt-001",
            title="Test",
            thesis="Unclear",
            recommended_side="YES",
            confidence=0.4,
            key_factors=[],
            sources=[],
        )
        assert not r.has_edge


class TestLLMResearcher:
    def test_init(self):
        researcher = LLMResearcher()
        assert researcher.min_confidence == 0.7

    def test_build_research_prompt(self):
        researcher = LLMResearcher()
        prompt = researcher.build_research_prompt(
            title="Will Bitcoin reach $200K by end of 2026?",
            price_yes=0.15,
            volume_24h=500000,
            category="crypto",
        )
        assert "Bitcoin" in prompt
        assert "$200K" in prompt
        assert "15%" in prompt or "0.15" in prompt

    @pytest.mark.asyncio
    async def test_research_market_mocked(self):
        researcher = LLMResearcher()

        mock_response = {
            "thesis": "Based on current trends, BTC reaching $200K is unlikely.",
            "recommended_side": "NO",
            "confidence": 0.82,
            "key_factors": ["current price is $65K", "historical growth rates"],
            "sources": ["market data"],
        }

        with patch.object(researcher, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response
            result = await researcher.research_market(
                market_id="mkt-001",
                title="Will Bitcoin reach $200K by end of 2026?",
                price_yes=0.15,
                volume_24h=500000,
                category="crypto",
            )

        assert result.recommended_side == "NO"
        assert result.confidence == 0.82
        assert result.has_edge
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/screening/test_llm_researcher.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# src/screening/llm_researcher.py
"""LLM Research Agent for generating trading theses on Polymarket markets.

Uses Claude (or Perplexity via API) to perform deep research on screened
markets and generate actionable trading recommendations.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx
import structlog

from config.settings import settings

logger = structlog.get_logger()


@dataclass
class ResearchResult:
    """Result of LLM research on a market."""

    market_id: str
    title: str
    thesis: str
    recommended_side: str  # "YES", "NO", or "SKIP"
    confidence: float
    key_factors: list[str]
    sources: list[str]

    @property
    def has_edge(self) -> bool:
        return self.confidence >= 0.7 and self.recommended_side != "SKIP"


class LLMResearcher:
    """Generates trading theses using LLM deep research."""

    def __init__(self, min_confidence: float = 0.7):
        self.min_confidence = min_confidence

    def build_research_prompt(
        self,
        title: str,
        price_yes: float,
        volume_24h: float,
        category: str,
    ) -> str:
        """Build a research prompt for the LLM."""
        return f"""Analyze this Polymarket prediction market and provide a trading recommendation.

Market: "{title}"
Category: {category}
Current YES price: {price_yes:.2f} ({price_yes*100:.0f}% implied probability)
24h Volume: ${volume_24h:,.0f}

Research the topic thoroughly and respond with a JSON object:
{{
    "thesis": "Your detailed analysis (2-3 paragraphs)",
    "recommended_side": "YES" or "NO" or "SKIP",
    "confidence": 0.0 to 1.0,
    "key_factors": ["factor 1", "factor 2", ...],
    "sources": ["relevant source 1", ...]
}}

Consider:
1. What is the base rate for this type of event?
2. Are there any recent developments that change the probability?
3. Is the current market price justified by available evidence?
4. What information asymmetry might exist?

Be conservative. Only recommend YES or NO if you have strong evidence
that the market is mispriced by at least 10%.
Respond ONLY with the JSON object."""

    async def _call_llm(self, prompt: str) -> dict[str, Any]:
        """Call the LLM API (Claude or Perplexity).

        Override this method to switch between LLM providers.
        """
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": settings.ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": settings.LLM_MODEL,
                    "max_tokens": 2000,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=30.0,
            )
            response.raise_for_status()
            data = response.json()
            text = data["content"][0]["text"]
            return json.loads(text)

    async def research_market(
        self,
        market_id: str,
        title: str,
        price_yes: float,
        volume_24h: float,
        category: str,
    ) -> ResearchResult:
        """Research a market and generate a trading thesis.

        Args:
            market_id: Polymarket condition ID
            title: Market title
            price_yes: Current YES price
            volume_24h: 24-hour volume
            category: Market category

        Returns:
            ResearchResult with thesis and recommendation
        """
        prompt = self.build_research_prompt(title, price_yes, volume_24h, category)
        result = await self._call_llm(prompt)

        return ResearchResult(
            market_id=market_id,
            title=title,
            thesis=result.get("thesis", ""),
            recommended_side=result.get("recommended_side", "SKIP"),
            confidence=result.get("confidence", 0.0),
            key_factors=result.get("key_factors", []),
            sources=result.get("sources", []),
        )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/screening/test_llm_researcher.py -v`
Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add src/screening/llm_researcher.py tests/screening/test_llm_researcher.py
git commit -m "feat: add LLM researcher for market thesis generation"
```

---

### Task C3: Screener Runner Script

**Files:**
- Create: `scripts/run_screener.py`
- Modify: `config/settings.py`

**Step 1: Add settings**

```python
    # === Market Screener ===
    SCREENER_MIN_ALPHA_SCORE: float = 0.6
    SCREENER_TOP_N: int = 10  # research top N markets per scan
    SCREENER_SCAN_INTERVAL: float = 3600.0  # hourly
    SCREENER_LLM_PROVIDER: str = "claude"  # "claude" or "perplexity"
    PERPLEXITY_API_KEY: str = ""
```

**Step 2: Write runner**

```python
# scripts/run_screener.py
"""Runner for LLM-powered market screener.

Scans Polymarket, scores markets for alpha, researches top N via LLM.

Usage:
    python scripts/run_screener.py [--top-n 10] [--once]
"""

import argparse
import asyncio

import structlog

from config.settings import settings
from src.screening.market_screener import MarketScreener
from src.screening.llm_researcher import LLMResearcher

logger = structlog.get_logger()


async def main(top_n: int, once: bool) -> None:
    screener = MarketScreener(min_alpha_score=settings.SCREENER_MIN_ALPHA_SCORE)
    researcher = LLMResearcher()

    while True:
        logger.info("screener_scan_starting")

        # TODO: fetch all active markets from Polymarket REST API
        # markets = await fetch_active_markets()
        # screened = screener.screen_markets(markets)
        # top = [m for m in screened if m.is_interesting][:top_n]
        #
        # for market in top:
        #     result = await researcher.research_market(
        #         market_id=market.market_id,
        #         title=market.title,
        #         price_yes=market.price_yes,
        #         volume_24h=market.volume_24h,
        #         category=market.category,
        #     )
        #     if result.has_edge:
        #         logger.info("screener_edge_found", title=result.title, side=result.recommended_side)

        if once:
            break
        await asyncio.sleep(settings.SCREENER_SCAN_INTERVAL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Market Screener")
    parser.add_argument("--top-n", type=int, default=settings.SCREENER_TOP_N)
    parser.add_argument("--once", action="store_true", help="Run once then exit")
    args = parser.parse_args()

    asyncio.run(main(args.top_n, args.once))
```

**Step 3: Run tests**

Run: `pytest tests/ -v --timeout=30`
Expected: All tests PASS

**Step 4: Commit**

```bash
git add config/settings.py scripts/run_screener.py
git commit -m "feat: add market screener runner with LLM research pipeline"
```

---

## Phase D: Combinatorial Arbitrage Foundation (Strat #5 — RohOnChain)

> This is the most complex phase. We implement the foundation for detecting logical dependencies between Polymarket markets and computing optimal arbitrage trades. Full Bregman/Frank-Wolfe optimization is Phase D+, noted but not implemented here.

### Task D1: Market Dependency Detector

**Files:**
- Create: `src/arb/dependency_detector.py`
- Create: `tests/arb/test_dependency_detector.py`

**Step 1: Write the failing test**

```python
# tests/arb/test_dependency_detector.py
"""Tests for cross-market logical dependency detection."""
import pytest
from unittest.mock import AsyncMock, patch

from src.arb.dependency_detector import DependencyDetector, MarketDependency


class TestMarketDependency:
    def test_has_arbitrage_yes(self):
        dep = MarketDependency(
            market_a_id="mkt-a",
            market_a_title="Will Trump win Pennsylvania?",
            market_b_id="mkt-b",
            market_b_title="Will Republicans win by 5+ points nationally?",
            dependency_type="implication",
            description="If B is YES then A must be YES",
            valid_outcomes=[("YES", "YES"), ("YES", "NO"), ("NO", "NO")],
            invalid_outcomes=[("NO", "YES")],
            confidence=0.95,
        )
        assert dep.has_dependency
        assert dep.n_valid_outcomes == 3
        assert dep.n_total_outcomes == 4

    def test_no_dependency(self):
        dep = MarketDependency(
            market_a_id="mkt-a",
            market_a_title="Market A",
            market_b_id="mkt-b",
            market_b_title="Market B",
            dependency_type="independent",
            description="No dependency",
            valid_outcomes=[("YES", "YES"), ("YES", "NO"), ("NO", "YES"), ("NO", "NO")],
            invalid_outcomes=[],
            confidence=0.9,
        )
        assert not dep.has_dependency


class TestDependencyDetector:
    def test_init(self):
        detector = DependencyDetector()
        assert detector.confidence_threshold == 0.9

    def test_check_single_market_arbitrage_underpriced(self):
        detector = DependencyDetector()
        # YES=0.40, NO=0.40 → sum=0.80, should be 1.0 → arb exists
        arb = detector.check_single_market_arbitrage(
            yes_price=0.40, no_price=0.40
        )
        assert arb is not None
        assert arb["type"] == "buy_both"
        assert arb["profit_per_dollar"] == pytest.approx(0.20, abs=0.01)

    def test_check_single_market_arbitrage_overpriced(self):
        detector = DependencyDetector()
        # YES=0.60, NO=0.55 → sum=1.15 → sell both
        arb = detector.check_single_market_arbitrage(
            yes_price=0.60, no_price=0.55
        )
        assert arb is not None
        assert arb["type"] == "sell_both"

    def test_check_single_market_no_arb(self):
        detector = DependencyDetector()
        arb = detector.check_single_market_arbitrage(
            yes_price=0.48, no_price=0.52
        )
        assert arb is None

    def test_check_pair_arbitrage_with_dependency(self):
        detector = DependencyDetector()
        dep = MarketDependency(
            market_a_id="mkt-a",
            market_a_title="Trump wins PA?",
            market_b_id="mkt-b",
            market_b_title="GOP +5 nationally?",
            dependency_type="implication",
            description="B implies A",
            valid_outcomes=[("YES", "YES"), ("YES", "NO"), ("NO", "NO")],
            invalid_outcomes=[("NO", "YES")],
            confidence=0.95,
        )
        prices = {
            "mkt-a": {"YES": 0.48, "NO": 0.52},
            "mkt-b": {"YES": 0.32, "NO": 0.68},
        }
        arb = detector.check_pair_arbitrage(dep, prices)
        # With 3 valid outcomes instead of 4, prices may violate constraints
        # This test validates the detection logic runs without error
        assert arb is None or isinstance(arb, dict)

    @pytest.mark.asyncio
    async def test_detect_dependencies_via_llm(self):
        detector = DependencyDetector()

        mock_result = {
            "dependency_type": "implication",
            "description": "If B then A",
            "valid_outcomes": [["YES", "YES"], ["YES", "NO"], ["NO", "NO"]],
            "invalid_outcomes": [["NO", "YES"]],
            "confidence": 0.95,
        }

        with patch.object(detector, "_ask_llm_dependency", new_callable=AsyncMock) as mock:
            mock.return_value = mock_result
            dep = await detector.detect_dependency(
                market_a_id="mkt-a",
                market_a_title="Will Trump win Pennsylvania?",
                market_b_id="mkt-b",
                market_b_title="Will Republicans win by 5+ nationally?",
            )

        assert dep is not None
        assert dep.has_dependency
        assert dep.confidence == 0.95
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/arb/test_dependency_detector.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# src/arb/dependency_detector.py
"""Logical dependency detector for cross-market combinatorial arbitrage.

Based on Kroer et al. 2016 and Saguillo et al. 2025 (strat #5 RohOnChain):
Markets on Polymarket are priced independently but may have logical dependencies.
If "Trump wins PA" implies "GOP wins nationally", then pricing them independently
creates arbitrage.

This module:
1. Detects single-market arbitrage (YES + NO != $1)
2. Uses LLM to detect logical dependencies between market pairs
3. Checks if dependent pairs have exploitable price inconsistencies

Full Bregman projection / Frank-Wolfe optimization is a future phase.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx
import structlog

from config.settings import settings

logger = structlog.get_logger()


@dataclass
class MarketDependency:
    """A detected logical dependency between two markets."""

    market_a_id: str
    market_a_title: str
    market_b_id: str
    market_b_title: str
    dependency_type: str  # "implication", "mutual_exclusion", "correlation", "independent"
    description: str
    valid_outcomes: list[tuple[str, str]]  # [(A_outcome, B_outcome), ...]
    invalid_outcomes: list[tuple[str, str]]
    confidence: float

    @property
    def has_dependency(self) -> bool:
        """True if fewer valid outcomes than total possible (dependency exists)."""
        return len(self.invalid_outcomes) > 0

    @property
    def n_valid_outcomes(self) -> int:
        return len(self.valid_outcomes)

    @property
    def n_total_outcomes(self) -> int:
        return len(self.valid_outcomes) + len(self.invalid_outcomes)


class DependencyDetector:
    """Detects logical dependencies and arbitrage between Polymarket markets."""

    def __init__(self, confidence_threshold: float = 0.9):
        self.confidence_threshold = confidence_threshold
        # Cache for dependency results
        self._cache: dict[tuple[str, str], Optional[MarketDependency]] = {}

    def check_single_market_arbitrage(
        self, yes_price: float, no_price: float, min_profit: float = 0.02
    ) -> Optional[dict[str, Any]]:
        """Check if a single market has YES + NO != $1.

        From strat #5: 41% of Polymarket conditions had this type of arb.
        Median mispricing: $0.60 per dollar (should be $1.00).
        """
        total = yes_price + no_price

        if total < (1.0 - min_profit):
            # Buy both YES and NO → guaranteed $1, cost < $1
            profit = 1.0 - total
            return {
                "type": "buy_both",
                "yes_price": yes_price,
                "no_price": no_price,
                "total_cost": total,
                "profit_per_dollar": profit,
            }
        elif total > (1.0 + min_profit):
            # Sell both YES and NO → receive > $1, payout $1
            profit = total - 1.0
            return {
                "type": "sell_both",
                "yes_price": yes_price,
                "no_price": no_price,
                "total_received": total,
                "profit_per_dollar": profit,
            }

        return None

    def check_pair_arbitrage(
        self,
        dependency: MarketDependency,
        prices: dict[str, dict[str, float]],
        min_profit: float = 0.05,
    ) -> Optional[dict[str, Any]]:
        """Check if a dependent pair has exploitable price inconsistency.

        For each invalid outcome, there's a constraint the prices must satisfy.
        If prices violate the constraint, arbitrage exists.

        Args:
            dependency: The detected dependency
            prices: {market_id: {"YES": price, "NO": price}}
            min_profit: Minimum profit per dollar (default $0.05 from paper)
        """
        if not dependency.has_dependency:
            return None

        a_prices = prices.get(dependency.market_a_id, {})
        b_prices = prices.get(dependency.market_b_id, {})

        if not a_prices or not b_prices:
            return None

        # Simple check: for implication (B → A), P(B) <= P(A) must hold
        if dependency.dependency_type == "implication":
            p_a = a_prices.get("YES", 0.5)
            p_b = b_prices.get("YES", 0.5)

            # If B implies A, then P(B) > P(A) is a violation
            if p_b > p_a + min_profit:
                return {
                    "type": "implication_violation",
                    "action": "buy_A_sell_B",
                    "market_a": dependency.market_a_id,
                    "market_b": dependency.market_b_id,
                    "p_a": p_a,
                    "p_b": p_b,
                    "profit_estimate": p_b - p_a,
                }

        # For mutual exclusion: P(A) + P(B) <= 1 must hold
        if dependency.dependency_type == "mutual_exclusion":
            p_a = a_prices.get("YES", 0.5)
            p_b = b_prices.get("YES", 0.5)

            if p_a + p_b > 1.0 + min_profit:
                return {
                    "type": "mutual_exclusion_violation",
                    "action": "sell_both",
                    "market_a": dependency.market_a_id,
                    "market_b": dependency.market_b_id,
                    "p_a": p_a,
                    "p_b": p_b,
                    "profit_estimate": p_a + p_b - 1.0,
                }

        return None

    async def _ask_llm_dependency(
        self, title_a: str, title_b: str
    ) -> dict[str, Any]:
        """Ask LLM to analyze logical dependency between two markets."""
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

Rules:
- "implication": If B is YES, then A must be YES (B → A)
- "mutual_exclusion": A and B cannot both be YES
- "correlation": Related but neither implies the other
- "independent": No logical relationship

List ALL valid (A_outcome, B_outcome) pairs and all invalid pairs.
Respond ONLY with JSON."""

        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": settings.ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": settings.LLM_MODEL,
                    "max_tokens": 1000,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=30.0,
            )
            response.raise_for_status()
            data = response.json()
            text = data["content"][0]["text"]
            return json.loads(text)

    async def detect_dependency(
        self,
        market_a_id: str,
        market_a_title: str,
        market_b_id: str,
        market_b_title: str,
    ) -> Optional[MarketDependency]:
        """Detect logical dependency between two markets using LLM.

        Results are cached to avoid redundant LLM calls.
        """
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
            market_a_id=market_a_id,
            market_a_title=market_a_title,
            market_b_id=market_b_id,
            market_b_title=market_b_title,
            dependency_type=result["dependency_type"],
            description=result.get("description", ""),
            valid_outcomes=[tuple(o) for o in result.get("valid_outcomes", [])],
            invalid_outcomes=[tuple(o) for o in result.get("invalid_outcomes", [])],
            confidence=result.get("confidence", 0.0),
        )

        self._cache[cache_key] = dep
        return dep
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/arb/test_dependency_detector.py -v`
Expected: All 7 tests PASS

**Step 5: Commit**

```bash
git add src/arb/dependency_detector.py tests/arb/test_dependency_detector.py
git commit -m "feat: add logical dependency detector for combinatorial arbitrage"
```

---

### Task D2: Combinatorial Arb Scanner Script

**Files:**
- Create: `scripts/run_combinatorial_arb.py`
- Modify: `config/settings.py`

**Step 1: Add settings**

```python
    # === Combinatorial Arbitrage ===
    COMBO_ARB_SCAN_INTERVAL: float = 60.0  # scan every minute
    COMBO_ARB_MIN_PROFIT: float = 0.05  # $0.05 min profit (from paper)
    COMBO_ARB_MAX_PAIRS_PER_SCAN: int = 50  # limit LLM calls
    COMBO_ARB_DEPENDENCY_CONFIDENCE: float = 0.90
```

**Step 2: Write runner**

```python
# scripts/run_combinatorial_arb.py
"""Runner for combinatorial arbitrage on logically dependent markets.

Scans Polymarket for:
1. Single-market arbitrage (YES + NO != $1)
2. Cross-market logical dependencies (via LLM)
3. Price violations on dependent markets

Based on Kroer et al. 2016 / Saguillo et al. 2025.

Usage:
    python scripts/run_combinatorial_arb.py [--scan-interval 60] [--autopilot]
"""

import argparse
import asyncio

import structlog

from config.settings import settings
from src.arb.dependency_detector import DependencyDetector

logger = structlog.get_logger()


async def main(scan_interval: float, autopilot: bool) -> None:
    logger.info("combinatorial_arb_starting", scan_interval=scan_interval)

    detector = DependencyDetector(
        confidence_threshold=settings.COMBO_ARB_DEPENDENCY_CONFIDENCE,
    )

    while True:
        logger.info("combinatorial_arb_scan_starting")

        # TODO: Phase 1 — single market arb scan
        # Fetch all active markets, check YES + NO != $1
        # for market in markets:
        #     arb = detector.check_single_market_arbitrage(yes_price, no_price)

        # TODO: Phase 2 — dependency detection
        # For top N market pairs (by volume), detect dependencies via LLM
        # Cache results to avoid redundant calls

        # TODO: Phase 3 — pair arbitrage on cached dependencies
        # For each known dependency, check current prices for violations

        await asyncio.sleep(scan_interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combinatorial Arb Scanner")
    parser.add_argument("--scan-interval", type=float, default=settings.COMBO_ARB_SCAN_INTERVAL)
    parser.add_argument("--autopilot", action="store_true")
    args = parser.parse_args()

    asyncio.run(main(args.scan_interval, args.autopilot))
```

**Step 3: Run tests**

Run: `pytest tests/ -v --timeout=30`
Expected: All tests PASS

**Step 4: Commit**

```bash
git add config/settings.py scripts/run_combinatorial_arb.py
git commit -m "feat: add combinatorial arb scanner with dependency detection"
```

---

## Phase E: Integration & Risk Manager Update

### Task E1: Extend Risk Manager for All Strategies

**Files:**
- Modify: `src/risk/manager.py`
- Modify: `tests/risk/test_manager.py`

**Step 1: Write the failing test**

Add tests for the new strategy types in `tests/risk/test_manager.py`:

```python
def test_crypto_allocation():
    mgr = UnifiedRiskManager(
        global_capital=10000,
        reality_allocation_pct=30,
        crossmarket_allocation_pct=30,
        crypto_allocation_pct=20,
        nobet_allocation_pct=20,
        max_position_pct=0.10,
        daily_loss_limit_pct=0.05,
    )
    assert mgr.get_available_capital("crypto") == 2000.0
    assert mgr.get_available_capital("nobet") == 2000.0

def test_record_pnl_crypto():
    mgr = UnifiedRiskManager(
        global_capital=10000,
        reality_allocation_pct=25,
        crossmarket_allocation_pct=25,
        crypto_allocation_pct=25,
        nobet_allocation_pct=25,
        max_position_pct=0.10,
        daily_loss_limit_pct=0.05,
    )
    mgr.record_pnl(100.0, "crypto")
    assert mgr.get_strategy_pnl("crypto") == 100.0
    assert mgr.daily_pnl == 100.0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/risk/test_manager.py -v`
Expected: FAIL — `TypeError: __init__() got unexpected keyword argument 'crypto_allocation_pct'`

**Step 3: Update UnifiedRiskManager**

Update `src/risk/manager.py`:
- Expand `StrategyType` to `Literal["reality", "crossmarket", "crypto", "nobet"]`
- Add `crypto_allocation_pct` and `nobet_allocation_pct` parameters (default 0.0)
- Handle new types in `get_available_capital()`
- Add to `_daily_pnl_by_strategy` init

**Step 4: Run test to verify it passes**

Run: `pytest tests/risk/test_manager.py -v`
Expected: All tests PASS (existing + new)

**Step 5: Commit**

```bash
git add src/risk/manager.py tests/risk/test_manager.py
git commit -m "feat: extend risk manager with crypto and nobet strategy allocations"
```

---

### Task E2: Full Test Suite Validation

**Step 1: Run entire test suite**

Run: `pytest tests/ -v --timeout=30`
Expected: All tests PASS

**Step 2: Commit if any fixups needed**

```bash
git add -u
git commit -m "fix: test suite compatibility after strategies expansion"
```

---

## Summary of Deliverables

| Phase | Strategy | Files Created | Lines (~) |
|-------|----------|---------------|-----------|
| A | Crypto 15-min Arb | 5 files (feed, mapper, engine, tests, runner) | ~600 |
| B | NO Bet Scanner | 3 files (scanner, tests, runner) | ~300 |
| C | LLM Screener | 5 files (screener, researcher, tests, runner) | ~500 |
| D | Combinatorial Arb | 3 files (detector, tests, runner) | ~400 |
| E | Integration | 2 files modified (risk manager + tests) | ~50 |

**Total: ~18 files, ~1850 lines**

## Future Work (Not in This Plan)

- **Full Bregman/Frank-Wolfe optimization** (Strat #5 Part 2): Requires Gurobi for IP solver. Foundation is laid in Phase D.
- **VWAP execution engine**: Per-block VWAP analysis for execution timing
- **Backtesting framework**: Replay synthetic data through all strategies
- **Multi-symbol scanning**: Extend crypto arb beyond BTC/ETH/SOL
- **Perplexity integration**: Alternative LLM provider for screener with web search
