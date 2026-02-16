#!/usr/bin/env python3
"""Passive time-decay maker for Polymarket 15-min crypto markets.

STRATEGY (validated by backtest_crypto_minute.py):
    On Polymarket 15-min crypto binary markets (BTC/ETH up/down), the
    favourite side (priced 0.75-0.85) wins ~1% more often than implied.
    Since Jan 2026, taker fees (~1% at p=0.80) eat this edge — but
    MAKER orders pay 0 fees and earn rebates.

    How it works:
      1. Subscribe to orderbooks for all active 15-min crypto markets.
      2. On each tick, check both outcomes. When the best BID on an
         outcome enters [target_bid, max_bid] (e.g. 0.75-0.85),
         place a GTC buy at that bid price. The order sits in the
         book as a maker order (1c below the ask).
      3. When filled, hold to market resolution (15 min).
      4. Win ~76-81% → net +1% edge per fill, 0 maker fees.

    This is a directional time-decay bet with maker execution, NOT the
    two-sided inventory pair-trade of run_crypto_maker.py.

USAGE:
    ./run run_crypto_td_maker.py --paper             # default (paper mode)
    ./run run_crypto_td_maker.py --live               # real orders
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import httpx
import structlog

from src.utils.logging import configure_logging

configure_logging()

try:
    import uvloop
except ImportError:
    uvloop = None

from config.settings import settings
from src.arb.polymarket_executor import PolymarketExecutor
from src.feeds.polymarket import PolymarketFeed, PolymarketUserFeed, UserTradeEvent

# Crypto market discovery and parsing utilities.
from src.feeds.chainlink import ChainlinkFeed
from src.utils.crypto_markets import SLUG_TO_CHAINLINK, fetch_crypto_markets
from src.utils.fair_value import estimate_fair_value
from src.utils.parsing import parse_json_list, _to_float, _first_event_slug
from src.execution import TradeManager, TradeIntent, FillResult
from src.risk.guard import RiskGuard

logger = structlog.get_logger()

TD_MAKER_EVENT_TYPE = "crypto_td_maker"


def compute_rung_prices(lo: float, hi: float, n_rungs: int) -> list[float]:
    """Compute evenly-spaced rung prices within [lo, hi], snapped to 1c."""
    if n_rungs <= 1:
        return [round((lo + hi) / 2, 2)]
    raw = [lo + i * (hi - lo) / (n_rungs - 1) for i in range(n_rungs)]
    seen: set[float] = set()
    result: list[float] = []
    for p in raw:
        rounded = round(p, 2)
        if rounded not in seen:
            seen.add(rounded)
            result.append(rounded)
    return result


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class PassiveOrder:
    """A GTC maker order waiting to be filled."""
    order_id: str
    condition_id: str
    outcome: str  # "Up" or "Down"
    token_id: str
    price: float
    size_usd: float
    placed_at: float
    cancelled_at: float = 0.0


@dataclass(slots=True)
class OpenPosition:
    """A filled passive order held until market resolution."""
    condition_id: str
    outcome: str
    token_id: str
    entry_price: float
    size_usd: float
    shares: float  # size_usd / price
    filled_at: float


# ---------------------------------------------------------------------------
# CryptoTDMaker
# ---------------------------------------------------------------------------

class CryptoTDMaker:
    """Passive time-decay maker for 15-min crypto binary markets.

    Watches orderbooks via WebSocket. When the bid on either outcome
    enters the target range [target_bid, max_bid], places a GTC buy
    at the bid price. Holds filled positions to market resolution.
    """

    def __init__(
        self,
        *,
        executor: Optional[PolymarketExecutor],
        polymarket: PolymarketFeed,
        user_feed: Optional[PolymarketUserFeed],
        manager: Optional[TradeManager] = None,
        symbols: list[str],
        target_bid: float = 0.75,
        max_bid: float = 0.85,
        order_size_usd: float = 10.0,
        max_total_exposure_usd: float = 200.0,
        paper_mode: bool = True,
        discovery_interval: float = 60.0,
        maker_loop_interval: float = 0.5,
        strategy_tag: str = "crypto_td_maker",
        guard: Optional[RiskGuard] = None,
        db_url: str = "",
        ladder_rungs: int = 1,
        min_move_pct: float = 0.0,
        max_move_pct: float = 0.0,
        min_entry_minutes: float = 0.0,
        max_entry_minutes: float = 0.0,
        chainlink_feed: Optional[ChainlinkFeed] = None,
        model_path: str = "",
        hybrid_skip_below: float = 0.55,
        hybrid_taker_above: float = 0.72,
        stoploss_peak: float = 0.0,
        stoploss_exit: float = 0.0,
        stoploss_fair_margin: float = 0.10,
        entry_fair_margin: float = 0.0,
        exit_model_path: str = "",
        exit_threshold: float = 0.35,
    ) -> None:
        self.executor = executor
        self.polymarket = polymarket
        self.user_feed = user_feed
        self.manager = manager
        self.chainlink_feed = chainlink_feed
        self.symbols = symbols
        self.target_bid = target_bid
        self.max_bid = max_bid
        self.order_size_usd = order_size_usd
        self.max_total_exposure_usd = max_total_exposure_usd
        self.paper_mode = paper_mode
        self.discovery_interval = discovery_interval
        self.maker_loop_interval = maker_loop_interval
        self.strategy_tag = strategy_tag
        self.guard = guard
        self._db_url = db_url
        self.ladder_rungs = ladder_rungs
        self.min_move_pct = min_move_pct
        self.max_move_pct = max_move_pct
        self.min_entry_minutes = min_entry_minutes
        self.max_entry_minutes = max_entry_minutes
        self.stoploss_peak = stoploss_peak
        self.stoploss_exit = stoploss_exit
        self.stoploss_fair_margin = stoploss_fair_margin
        self.entry_fair_margin = entry_fair_margin
        self.rung_prices = compute_rung_prices(target_bid, max_bid, ladder_rungs)
        self._last_book_update: float = time.time()

        # State
        self.known_markets: dict[str, dict[str, Any]] = {}  # cid -> raw market
        self.market_tokens: dict[tuple[str, str], str] = {}  # (cid, outcome) -> token_id
        self.market_outcomes: dict[str, list[str]] = {}  # cid -> ["Up", "Down"]
        self.active_orders: dict[str, PassiveOrder] = {}  # order_id -> order
        self._orders_by_cid_outcome: dict[tuple[str, str], str] = {}
        self.positions: dict[str, OpenPosition] = {}  # cid -> position
        # Orders being cancelled — kept for late fill detection.
        self._pending_cancels: dict[str, PassiveOrder] = {}  # order_id -> order
        self._http_client: Optional[httpx.AsyncClient] = None
        self._cycle_count: int = 0
        self._last_status_time: float = 0.0
        # Last known bid per (cid, outcome) — fallback for settlement when book empties.
        self._last_bids: dict[tuple[str, str], float] = {}
        # Map condition_id -> order_id for DB settlement tracking.
        self._position_order_ids: dict[str, str] = {}
        # Per-cid fill count — blocks re-ordering when >= ladder_rungs.
        self._cid_fill_count: dict[str, int] = {}  # cid -> number of fills
        # Rung-level dedup: (cid, outcome, price_cents) of placed/active rungs.
        self._rung_placed: set[tuple[str, str, int]] = set()

        # Chainlink reference prices for min-move filter.
        self._ref_prices: dict[str, float] = {}  # cid -> chainlink price at slot start
        self._cid_chainlink_symbol: dict[str, str] = {}  # cid -> "btc/usd"
        self._cid_slot_ts: dict[str, int] = {}  # cid -> slot start unix timestamp
        self._cid_fill_analytics: dict[str, dict] = {}  # cid -> {dir_move_pct, minutes_into_slot}
        self._last_p_win: dict[tuple[str, str], float] = {}  # (cid, outcome) -> model P(win)
        self._last_order_type: dict[tuple[str, str], str] = {}  # (cid, outcome) -> "maker"/"taker"

        # Book history ring buffer for trend features (per cid).
        # Each entry: (timestamp, bid_up, ask_up, spread_up, spread_down)
        self._book_history: dict[str, deque] = {}  # cid -> deque of tuples (max 20)

        # Stop-loss: track highest bid seen per position.
        self._position_bid_max: dict[str, float] = {}  # cid -> highest bid since fill
        self._position_last_bid: dict[str, float] = {}  # cid -> most recent non-None bid

        # Stats
        self.total_fills: int = 0
        self.total_wins: int = 0
        self.total_losses: int = 0
        self.realized_pnl: float = 0.0
        self.stoploss_overrides: int = 0  # fair value override count

        # ML entry model (optional)
        self._model = None
        self._model_features: list[str] = []
        if model_path:
            self._load_model(model_path)
        self.HYBRID_SKIP_BELOW = hybrid_skip_below
        self.HYBRID_TAKER_ABOVE = hybrid_taker_above

        # ML exit model (optional — complements or replaces rule-based stop-loss)
        self._exit_model = None
        self._exit_model_features: list[str] = []
        self._exit_threshold = exit_threshold
        if exit_model_path:
            self._load_exit_model(exit_model_path)

    def _load_model(self, path: str) -> None:
        """Load trained XGBoost entry model from joblib file."""
        import joblib
        payload = joblib.load(path)
        self._model = payload["model"]
        self._model_features = payload["feature_cols"]
        logger.info("entry_model_loaded", path=path,
                    features=len(self._model_features))

    def _load_exit_model(self, path: str) -> None:
        """Load trained XGBoost exit model from joblib file."""
        import joblib
        payload = joblib.load(path)
        self._exit_model = payload["model"]
        self._exit_model_features = payload["feature_cols"]
        logger.info("exit_model_loaded", path=path,
                    features=len(self._exit_model_features))

    def _record_book_snapshot(self, cid: str) -> None:
        """Append current book state to the history ring buffer for trend features."""
        bid_up, _, ask_up, _ = self.polymarket.get_best_levels(cid, "Up")
        if bid_up is None:
            return
        bid_down, _, ask_down, _ = self.polymarket.get_best_levels(cid, "Down")
        spread_up = (ask_up - bid_up) if (ask_up and bid_up) else 0.0
        spread_down = (ask_down - bid_down) if (ask_down and bid_down) else 0.0
        entry = (time.time(), bid_up, ask_up or 0.0, spread_up, spread_down)
        if cid not in self._book_history:
            self._book_history[cid] = deque(maxlen=20)
        self._book_history[cid].append(entry)

    def _get_trend(self, cid: str, seconds_ago: float, field_idx: int) -> float:
        """Look back in book history and return delta (current - past) for a field.

        field_idx: 1=bid_up, 2=ask_up, 3=spread_up, 4=spread_down
        Returns 0.0 if insufficient history.
        """
        hist = self._book_history.get(cid)
        if not hist or len(hist) < 2:
            return 0.0
        now_ts = hist[-1][0]
        current_val = hist[-1][field_idx]
        target_ts = now_ts - seconds_ago
        # Find closest snapshot to target_ts
        best = None
        best_diff = float("inf")
        for entry in hist:
            diff = abs(entry[0] - target_ts)
            if diff < best_diff:
                best_diff = diff
                best = entry
        if best is None or best_diff > seconds_ago * 0.8:
            return 0.0
        return current_val - best[field_idx]

    def _build_features(self, cid: str, outcome: str, bid: float) -> Optional[dict]:
        """Build feature dict matching training features from current market state."""
        if not self.chainlink_feed:
            return None

        sym = self._cid_chainlink_symbol.get(cid)
        slot_ts = self._cid_slot_ts.get(cid)
        if not sym or slot_ts is None:
            return None

        now = time.time()
        minutes = (now - slot_ts) / 60

        # Book levels for both outcomes
        bid_up, _, ask_up, _ = self.polymarket.get_best_levels(cid, "Up")
        bid_down, _, ask_down, _ = self.polymarket.get_best_levels(cid, "Down")
        bsz_up, asz_up = 0.0, 0.0
        bsz_down, asz_down = 0.0, 0.0

        # Try to get sizes from the book
        for out in ("Up", "Down"):
            b, bs, a, as_ = self.polymarket.get_best_levels(cid, out)
            if out == "Up":
                bsz_up = bs or 0.0
                asz_up = as_ or 0.0
            else:
                bsz_down = bs or 0.0
                asz_down = as_ or 0.0

        spread_up = (ask_up - bid_up) if (ask_up and bid_up) else 0.0
        spread_down = (ask_down - bid_down) if (ask_down and bid_down) else 0.0

        # Chainlink move
        dir_move = self._get_dir_move(cid, "Up")  # always Up direction
        ref = self._ref_prices.get(cid)
        current = self.chainlink_feed.get_price(sym)
        abs_move = abs((current - ref) / ref * 100) if (ref and current) else 0.0

        bid_up_val = bid_up or 0.0
        bid_down_val = bid_down or 0.0
        dt = datetime.fromtimestamp(now, tz=timezone.utc)

        features = {
            "bid_up": bid_up_val,
            "ask_up": ask_up or 0.0,
            "bid_down": bid_down_val,
            "ask_down": ask_down or 0.0,
            "spread_up": spread_up,
            "spread_down": spread_down,
            "bid_size_up": bsz_up,
            "ask_size_up": asz_up,
            "bid_size_down": bsz_down,
            "ask_size_down": asz_down,
            "dir_move_pct": dir_move or 0.0,
            "abs_move_pct": abs_move,
            "minutes_into_slot": minutes,
            "hour_utc": dt.hour,
            "day_of_week": dt.weekday(),
            # Derived (same as train_td_model.py)
            "spread_ratio": spread_up / max(spread_down, 0.001),
            "bid_imbalance": bid_up_val - bid_down_val,
            "fav_bid": max(bid_up_val, bid_down_val),
            "move_velocity": (dir_move or 0.0) / max(minutes, 0.5),
            "book_pressure": (bsz_up - bsz_down) / max(bsz_up + bsz_down, 0.01),
            # Trend features (from ring buffer)
            "bid_trend_30s": self._get_trend(cid, 30, 1),
            "bid_trend_2m": self._get_trend(cid, 120, 1),
            "ask_trend_30s": self._get_trend(cid, 30, 2),
            "spread_trend": self._get_trend(cid, 60, 3),
        }
        return features

    # Hybrid thresholds (set from CLI args in __init__)
    HYBRID_SKIP_BELOW: float = 0.55
    HYBRID_TAKER_ABOVE: float = 0.72

    def _check_model(self, cid: str, outcome: str, bid: float
                     ) -> Optional[float]:
        """Run ML model inference. Returns P(win) or None if model unavailable.

        Also stores the recommended order type in ``_last_order_type``:
        - "skip" if p_win < HYBRID_SKIP_BELOW
        - "taker" if p_win >= HYBRID_TAKER_ABOVE
        - "maker" otherwise
        """
        if not self._model:
            return None
        features = self._build_features(cid, outcome, bid)
        if features is None:
            return None
        import numpy as np
        row = np.array([[features.get(f, 0.0) for f in self._model_features]])
        proba = self._model.predict_proba(row)[0]
        # proba[1] = P(Up), proba[0] = P(Down)
        p_win = proba[1] if outcome == "Up" else proba[0]
        p_win = float(p_win)

        if p_win < self.HYBRID_SKIP_BELOW:
            self._last_order_type[(cid, outcome)] = "skip"
        elif p_win >= self.HYBRID_TAKER_ABOVE:
            self._last_order_type[(cid, outcome)] = "taker"
        else:
            self._last_order_type[(cid, outcome)] = "maker"

        return p_win

    # ------------------------------------------------------------------
    # DB persistence
    # ------------------------------------------------------------------

    async def _load_db_state(self) -> None:
        """Restore active orders and positions from DB on startup.

        Stale rows (older than 30 min) are cleaned up — their 15-min markets
        have already resolved, so loading them would create orphan positions
        that settle with unknown resolution.
        """
        if not self._db_url:
            return
        from src.db.td_orders import load_orders, delete_order
        rows = await load_orders(
            db_url=self._db_url, platform="polymarket",
            strategy_tag=self.strategy_tag,
        )
        now = time.time()
        stale_cutoff = now - 1800  # 30 min
        stale_count = 0
        for row in rows:
            # Skip and clean up stale orders from previous sessions.
            row_ts = row.filled_at or row.placed_at or 0.0
            if row_ts > 0 and row_ts < stale_cutoff:
                stale_count += 1
                self._db_fire(delete_order(db_url=self._db_url, order_id=row.order_id))
                continue

            if row.status == "pending":
                order = PassiveOrder(
                    order_id=row.order_id, condition_id=row.condition_id,
                    outcome=row.outcome, token_id=row.token_id,
                    price=row.price, size_usd=row.size_usd,
                    placed_at=row.placed_at or 0.0,
                )
                self.active_orders[row.order_id] = order
                self._orders_by_cid_outcome[(row.condition_id, row.outcome)] = row.order_id
            elif row.status == "filled":
                pos = OpenPosition(
                    condition_id=row.condition_id, outcome=row.outcome,
                    token_id=row.token_id, entry_price=row.price,
                    size_usd=row.size_usd,
                    shares=row.shares or row.size_usd / row.price,
                    filled_at=row.filled_at or 0.0,
                )
                self.positions[row.condition_id] = pos
                self._position_order_ids[row.condition_id] = row.order_id
                self._position_bid_max[row.condition_id] = row.price
        if stale_count:
            logger.info("td_stale_orders_cleaned", count=stale_count)

        # Restore ladder state from loaded DB rows.
        for cid, pos in self.positions.items():
            # Each restored filled position counts as one ladder fill.
            self._cid_fill_count[cid] = self._cid_fill_count.get(cid, 0) + 1
        for oid, order in self.active_orders.items():
            # Restore rung dedup for pending orders.
            self._rung_placed.add(
                (order.condition_id, order.outcome, int(round(order.price * 100))))

        if self.active_orders or self.positions:
            logger.info(
                "td_db_state_loaded",
                orders=len(self.active_orders),
                positions=len(self.positions),
                fill_counts=dict(self._cid_fill_count),
            )

    def _db_fire(self, coro) -> None:
        """Schedule a DB coroutine as fire-and-forget."""
        if not self._db_url:
            return
        try:
            asyncio.get_running_loop().create_task(coro)
        except RuntimeError:
            pass

    async def _db_save_order(self, order: PassiveOrder) -> None:
        from src.db.td_orders import save_order
        await save_order(
            db_url=self._db_url, platform="polymarket",
            strategy_tag=self.strategy_tag, order_id=order.order_id,
            condition_id=order.condition_id, token_id=order.token_id,
            outcome=order.outcome, price=order.price,
            size_usd=order.size_usd, status="pending",
            placed_at=order.placed_at,
            extra={},
        )

    async def _db_mark_filled(self, order_id: str, shares: float, filled_at: float) -> None:
        from src.db.td_orders import mark_filled
        await mark_filled(
            db_url=self._db_url, order_id=order_id,
            shares=shares, filled_at=filled_at,
        )

    async def _db_delete_order(self, order_id: str) -> None:
        from src.db.td_orders import delete_order
        await delete_order(db_url=self._db_url, order_id=order_id)

    async def _db_mark_settled(self, order_id: str, pnl: float, settled_at: float) -> None:
        from src.db.td_orders import mark_settled
        await mark_settled(
            db_url=self._db_url, order_id=order_id,
            pnl=pnl, settled_at=settled_at,
        )

    # ------------------------------------------------------------------
    # Chainlink price helpers (min-move filter)
    # ------------------------------------------------------------------

    def _get_dir_move(self, cid: str, outcome: str) -> Optional[float]:
        """Directional move of underlying vs slot open (%), positive = in bet direction.

        Returns None when data is unavailable.
        """
        ref = self._ref_prices.get(cid)
        sym = self._cid_chainlink_symbol.get(cid)
        if not ref or not sym or not self.chainlink_feed:
            return None
        current = self.chainlink_feed.get_price(sym)
        if current is None:
            return None
        move_pct = (current - ref) / ref * 100
        return -move_pct if outcome == "Down" else move_pct

    def _check_min_move(self, cid: str, outcome: str, price: float = 0.0) -> bool:
        """Check if underlying has moved enough in the bet direction.

        When *price* is provided, the threshold scales with loss:win asymmetry.
        At 0.85 one loss wipes ~5.7 wins vs ~3 at 0.75, so expensive rungs
        demand proportionally more confirmation.

        Returns True (allow) when min_move_pct is 0 or data is unavailable.
        """
        if self.min_move_pct <= 0:
            return True
        move = self._get_dir_move(cid, outcome)
        if move is None:
            return True
        if price > 0 and price < 1.0:
            base_ratio = self.target_bid / (1 - self.target_bid)
            rung_ratio = price / (1 - price)
            threshold = self.min_move_pct * (rung_ratio / base_ratio)
        else:
            threshold = self.min_move_pct
        return move >= threshold

    def _check_max_move(self, cid: str, outcome: str) -> bool:
        """Check if underlying hasn't moved too much (excessive volatility).

        Returns True (allow) when max_move_pct is 0 or data is unavailable.
        """
        if self.max_move_pct <= 0:
            return True
        move = self._get_dir_move(cid, outcome)
        if move is None:
            return True
        return move <= self.max_move_pct

    def _check_fair_value_entry(self, cid: str, outcome: str,
                                price: float, now: float) -> bool:
        """Check if entry price is consistent with Chainlink fair value.

        Returns True (allow) when data is unavailable or entry_fair_margin is 0.
        Blocks entry if price exceeds fair value by more than entry_fair_margin.
        """
        if self.entry_fair_margin <= 0:
            return True
        fair = self._estimate_fair_value(cid, outcome, now)
        if fair is None:
            return True
        if price > fair + self.entry_fair_margin:
            logger.info("fair_value_entry_blocked", cid=cid[:16],
                        outcome=outcome, price=price, fair=round(fair, 3),
                        max_price=round(fair + self.entry_fair_margin, 3))
            return False
        return True

    def _estimate_fair_value(self, cid: str, outcome: str, now: float) -> Optional[float]:
        """Estimate fair value from Chainlink price and time remaining."""
        dir_move = self._get_dir_move(cid, outcome)
        slot_ts = self._cid_slot_ts.get(cid)
        if dir_move is None or slot_ts is None:
            return None
        minutes_into = (now - slot_ts) / 60
        minutes_remaining = max(15.0 - minutes_into, 1.0)
        return estimate_fair_value(dir_move, minutes_remaining)

    def _check_min_entry_time(self, cid: str) -> bool:
        """Return True if enough time has elapsed since slot start.

        Returns True (allow) when min_entry_minutes is 0 or slot timestamp
        is unavailable.
        """
        if self.min_entry_minutes <= 0:
            return True
        slot_ts = self._cid_slot_ts.get(cid)
        if slot_ts is None:
            return True
        elapsed = (time.time() - slot_ts) / 60
        return elapsed >= self.min_entry_minutes

    def _check_max_entry_time(self, cid: str) -> bool:
        """Return True if not too late into the slot.

        Returns True (allow) when max_entry_minutes is 0 or slot timestamp
        is unavailable.
        """
        if self.max_entry_minutes <= 0:
            return True
        slot_ts = self._cid_slot_ts.get(cid)
        if slot_ts is None:
            return True
        elapsed = (time.time() - slot_ts) / 60
        return elapsed <= self.max_entry_minutes

    # ------------------------------------------------------------------
    # Stop-loss: exit when bid peaked then crashed
    # ------------------------------------------------------------------

    async def _check_stop_losses(self, now: float) -> None:
        """Check all positions for stop-loss trigger and exit if needed.

        Two modes:
        1. Rule-based: bid_max >= stoploss_peak AND current bid <= stoploss_exit.
        2. ML exit model: P(win) < exit_threshold.
        ML takes priority when available.
        """
        exits: list[str] = []
        for cid, pos in self.positions.items():
            bid, _, _, _ = self.polymarket.get_best_levels(cid, pos.outcome)

            # Empty book: use last known bid to decide.
            # If last bid was already below stoploss_exit, the position was
            # in trouble before the book emptied — trigger stoploss.
            # Otherwise skip (momentary websocket gap).
            if bid is None:
                last_bid = self._position_last_bid.get(cid)
                prev_max = self._position_bid_max.get(cid, pos.entry_price)
                if (last_bid is not None
                        and last_bid <= self.stoploss_exit
                        and prev_max >= self.stoploss_peak):
                    fair = self._estimate_fair_value(cid, pos.outcome, now)
                    if fair is not None and fair > self.stoploss_exit + self.stoploss_fair_margin:
                        self.stoploss_overrides += 1
                        logger.info("stoploss_chainlink_override_empty_book",
                                    cid=cid[:16], last_bid=last_bid,
                                    fair=round(fair, 3))
                        continue
                    logger.warning("stoploss_bid_none_trigger", cid=cid[:16],
                                   last_bid=last_bid, bid_max=round(prev_max, 3),
                                   fair=round(fair, 3) if fair else None,
                                   peak=self.stoploss_peak, exit=self.stoploss_exit)
                    exits.append(cid)
                continue

            # Track last known bid and update bid max.
            self._position_last_bid[cid] = bid
            prev_max = self._position_bid_max.get(cid, pos.entry_price)
            if bid > prev_max:
                self._position_bid_max[cid] = bid
                prev_max = bid

            # ML exit model takes priority.
            if self._exit_model:
                p_win = self._check_exit_model(cid, pos, bid, now)
                if p_win is not None and p_win < self._exit_threshold:
                    logger.info("ml_exit_triggered", cid=cid[:16],
                                p_win=round(p_win, 3), threshold=self._exit_threshold,
                                bid=bid, bid_max=round(prev_max, 3))
                    exits.append(cid)
                continue  # skip rule-based when ML is active

            # Rule-based trigger.
            if prev_max >= self.stoploss_peak and bid <= self.stoploss_exit:
                # Fair value override: skip if Chainlink says position is still good.
                fair = self._estimate_fair_value(cid, pos.outcome, now)
                if fair is not None and fair > self.stoploss_exit + self.stoploss_fair_margin:
                    self.stoploss_overrides += 1
                    logger.info("stoploss_chainlink_override",
                                cid=cid[:16], bid=bid, fair=round(fair, 3),
                                exit_threshold=self.stoploss_exit)
                    continue
                logger.info("stoploss_rule_trigger", cid=cid[:16],
                            bid=bid, bid_max=round(prev_max, 3),
                            fair=round(fair, 3) if fair else None,
                            peak=self.stoploss_peak, exit=self.stoploss_exit)
                exits.append(cid)

        for cid in exits:
            pos = self.positions.get(cid)
            if not pos:
                continue
            await self._execute_stop_loss(pos, now)

    def _check_exit_model(self, cid: str, pos: OpenPosition,
                          bid: float, now: float) -> Optional[float]:
        """Run ML exit model inference. Returns P(win) or None."""
        features = self._build_features(cid, pos.outcome, bid)
        if features is None:
            return None

        # Add exit-specific features.
        bid_max = self._position_bid_max.get(cid, pos.entry_price)
        slot_ts = self._cid_slot_ts.get(cid)
        minutes_into = (now - slot_ts) / 60 if slot_ts else 0
        fill_minutes = (pos.filled_at - slot_ts) / 60 if slot_ts else 0

        features["entry_price"] = pos.entry_price
        features["bid_max"] = bid_max
        features["bid_drop"] = bid_max - bid
        features["bid_drop_pct"] = (bid_max - bid) / max(bid_max, 0.01)
        features["minutes_remaining"] = max(15.0 - minutes_into, 0)
        features["minutes_held"] = minutes_into - fill_minutes
        features["pnl_unrealized"] = bid - pos.entry_price

        import numpy as np
        row = np.array([[features.get(f, 0.0) for f in self._exit_model_features]])
        proba = self._exit_model.predict_proba(row)[0]
        p_win = float(proba[1] if pos.outcome == "Up" else proba[0])
        return p_win

    async def _execute_stop_loss(self, pos: OpenPosition, now: float) -> None:
        """Sell position at market (stop-loss early exit)."""
        bid, _, _, _ = self.polymarket.get_best_levels(pos.condition_id, pos.outcome)
        bid_max = self._position_bid_max.get(pos.condition_id, pos.entry_price)

        # Re-check: if bid recovered above stoploss_exit, abort the sell.
        # This prevents false triggers from momentary empty books.
        if bid is not None and bid > self.stoploss_exit:
            logger.info("stoploss_abort_recovered", cid=pos.condition_id[:16],
                        bid=bid, exit_threshold=self.stoploss_exit)
            return

        sell_price = bid if bid is not None else self.stoploss_exit

        # PnL: we sell shares at sell_price instead of waiting for resolution.
        pnl = pos.shares * (sell_price - pos.entry_price)

        # Live mode: sell FIRST, then clean up tracking only on success.
        sell_ok = self.paper_mode  # paper always "succeeds"
        if not self.paper_mode and self.manager:
            slug = _first_event_slug(self.known_markets.get(pos.condition_id, {}))
            # GTC at 0.01 + force_taker: sells at any bid, allows partial fills,
            # and bypasses post_only so the order crosses the spread immediately.
            sell_intent = TradeIntent(
                condition_id=pos.condition_id,
                token_id=pos.token_id,
                outcome=pos.outcome,
                side="SELL",
                price=0.01,
                size_usd=pos.shares * (1.0 - 0.01),
                reason="stop_loss_exit",
                title=slug,
                timestamp=now,
            )
            try:
                pending = await self.manager.place(
                    sell_intent, order_type="GTC", force_taker=True)
                sell_ok = bool(pending.order_id)  # empty string = executor error
            except Exception as exc:
                logger.error("stop_loss_sell_failed",
                             cid=pos.condition_id[:16], error=str(exc)[:80])

        if not sell_ok:
            logger.warning("stop_loss_sell_not_confirmed", cid=pos.condition_id[:16],
                           bid=bid, sell_price=sell_price)
            return  # keep position in tracking, will retry next tick

        # Remove position from all tracking.
        self.positions.pop(pos.condition_id, None)
        self._position_bid_max.pop(pos.condition_id, None)
        self._position_last_bid.pop(pos.condition_id, None)

        # Cancel any remaining orders for this market.
        oids_to_cancel = [oid for oid, o in self.active_orders.items()
                          if o.condition_id == pos.condition_id]
        for oid in oids_to_cancel:
            order = self.active_orders.pop(oid)
            self._orders_by_cid_outcome.pop((order.condition_id, order.outcome), None)
            self._rung_placed.discard(
                (order.condition_id, order.outcome, int(round(order.price * 100))))
            if not self.paper_mode and self.executor:
                asyncio.create_task(self._async_cancel(oid))
            else:
                self._db_fire(self._db_delete_order(oid))

        self.total_losses += 1
        self.realized_pnl += pnl

        # DB settlement.
        oid = self._position_order_ids.pop(pos.condition_id, None)
        if oid:
            self._db_fire(self._db_mark_settled(oid, pnl, now))

        fair = self._estimate_fair_value(pos.condition_id, pos.outcome, now)
        logger.info(
            "td_stop_loss_exit",
            condition_id=pos.condition_id[:16],
            outcome=pos.outcome,
            entry=pos.entry_price,
            sell=sell_price,
            bid_max=round(bid_max, 3),
            pnl=round(pnl, 4),
            shares=round(pos.shares, 2),
            fair_value=round(fair, 3) if fair else None,
            paper=self.paper_mode,
        )

        # Telegram notification via TradeManager.
        if self.manager:
            slug = _first_event_slug(self.known_markets.get(pos.condition_id, {}))
            settle_intent = TradeIntent(
                condition_id=pos.condition_id,
                token_id=pos.token_id,
                outcome=pos.outcome,
                side="SELL",
                price=pos.entry_price,
                size_usd=pos.size_usd,
                reason="stop_loss_exit",
                title=slug,
                timestamp=now,
            )
            settle_fill = FillResult(
                filled=True,
                shares=pos.shares,
                avg_price=sell_price,
                pnl_delta=pnl,
            )
            context = f"STOP-LOSS | peak {bid_max:.2f} -> {sell_price:.2f}"
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self.manager.record_settle_direct(
                    settle_intent, settle_fill,
                    extra_state={"stop_loss": True, "bid_max": bid_max},
                    notify_context=context,
                ))
            except RuntimeError:
                pass

    @staticmethod
    def _parse_slug_info(slug: str) -> Optional[tuple[str, int]]:
        """Parse slug like 'btc-updown-15m-1771079400' → ('btc/usd', slot_ts)."""
        parts = slug.split("-")
        if len(parts) >= 4 and parts[-1].isdigit():
            chainlink_sym = SLUG_TO_CHAINLINK.get(parts[0].lower())
            if chainlink_sym:
                return chainlink_sym, int(parts[-1])
        return None

    # ------------------------------------------------------------------
    # Market discovery
    # ------------------------------------------------------------------

    async def market_discovery_loop(self) -> None:
        while True:
            try:
                await self._discover_markets()
            except Exception as exc:
                logger.error("discovery_error", error=str(exc))
            await asyncio.sleep(self.discovery_interval)

    async def _discover_markets(self) -> None:
        if self._http_client is None:
            return

        raw_markets = await fetch_crypto_markets(
            client=self._http_client,
            symbols=self.symbols,
        )

        new_count = 0
        new_user_markets: list[str] = []
        for mkt in raw_markets:
            cid = str(mkt.get("conditionId", ""))
            if not cid or cid in self.known_markets:
                continue

            outcomes = [str(o) for o in parse_json_list(mkt.get("outcomes", []))]
            clob_ids = [str(t) for t in parse_json_list(mkt.get("clobTokenIds", []))]
            if len(outcomes) < 2 or len(clob_ids) < 2:
                continue

            self.known_markets[cid] = mkt
            self.market_outcomes[cid] = outcomes[:2]
            token_map: dict[str, str] = {}
            for idx, outcome in enumerate(outcomes[:2]):
                self.market_tokens[(cid, outcome)] = clob_ids[idx]
                token_map[outcome] = clob_ids[idx]

            # Register tokens locally (don't send WS message yet).
            try:
                await self.polymarket.subscribe_market(
                    cid, token_map=token_map, send=False,
                )
            except Exception as exc:
                logger.warning("ws_subscribe_failed", condition_id=cid, error=str(exc))

            new_user_markets.append(cid)
            new_count += 1

        # Send ONE batched subscription for all new tokens at once,
        # avoiding rapid-fire replace messages that cause dropped snapshots.
        if new_count:
            try:
                await self.polymarket.flush_subscriptions()
            except Exception as exc:
                logger.warning("ws_flush_failed", error=str(exc))

            if self.user_feed and self.user_feed.is_connected:
                try:
                    await self.user_feed.subscribe_markets(new_user_markets)
                except Exception:
                    pass

            # Parse slot timestamps for all new markets (used for fill timing).
            for cid in new_user_markets:
                slug = _first_event_slug(self.known_markets.get(cid, {}))
                info = self._parse_slug_info(slug)
                if info:
                    chainlink_sym, slot_ts = info
                    self._cid_slot_ts[cid] = slot_ts
                    # Snapshot Chainlink prices (used for min-move filter + dir_move analytics).
                    if self.chainlink_feed:
                        self._cid_chainlink_symbol[cid] = chainlink_sym
                        ref = self.chainlink_feed.snapshot_price(chainlink_sym)
                        if ref:
                            self._ref_prices[cid] = ref
                            logger.info(
                                "td_ref_price_set",
                                cid=cid[:16],
                                symbol=chainlink_sym,
                                ref_price=ref,
                            )

            logger.info(
                "td_markets_discovered",
                new=new_count,
                total=len(self.known_markets),
                active_orders=len(self.active_orders),
                positions=len(self.positions),
            )

        # Prune expired markets.
        await self._prune_expired()

    # ------------------------------------------------------------------
    # Maker loop — the core: watch books, place bids in target range
    # ------------------------------------------------------------------

    async def maker_loop(self) -> None:
        """Event-driven loop: wake on book update or timeout."""
        await asyncio.sleep(3.0)  # wait for initial discovery + WS data

        while True:
            try:
                await asyncio.wait_for(
                    self.polymarket.book_updated.wait(),
                    timeout=self.maker_loop_interval,
                )
            except asyncio.TimeoutError:
                pass
            self.polymarket.book_updated.clear()

            try:
                await self._maker_tick()
            except Exception as exc:
                logger.error("maker_tick_error", error=str(exc))

    async def _maker_tick(self) -> None:
        now = time.time()
        self._cycle_count += 1

        # Sync book freshness from the feed (not gated behind guard).
        feed_ts = self.polymarket.last_update_ts
        if feed_ts > self._last_book_update:
            self._last_book_update = feed_ts

        # Expire stale pending cancels (>30s since cancel was sent).
        stale_cancel_ids = [
            oid for oid, order in self._pending_cancels.items()
            if order.cancelled_at > 0 and now - order.cancelled_at > 30.0
        ]
        for oid in stale_cancel_ids:
            order = self._pending_cancels.pop(oid)
            self._db_fire(self._db_delete_order(oid))
            logger.debug("td_pending_cancel_expired", order_id=oid[:16])

        # Circuit breaker gate
        if self.guard:
            await self.guard.heartbeat()
            if not await self.guard.is_trading_allowed(last_book_update=self._last_book_update):
                # Stale escalation: cancel all live orders to avoid adverse fills.
                if self.guard.should_cancel_orders and self.active_orders:
                    await self._cancel_all_orders("stale_escalation")
                # Still check paper fills and settle — just don't place new orders
                if self.paper_mode:
                    self._check_fills_paper(now)
                return

        # Check for paper fills every tick.
        if self.paper_mode:
            self._check_fills_paper(now)

        # Stop-loss: exit positions where bid peaked then crashed.
        if self.stoploss_peak > 0:
            await self._check_stop_losses(now)

        # Check exposure budget.
        current_exposure = sum(p.size_usd for p in self.positions.values())
        pending_exposure = sum(o.size_usd for o in self.active_orders.values())
        budget_left = self.max_total_exposure_usd - current_exposure - pending_exposure

        cancel_ids: list[str] = []
        # (cid, outcome, token_id, price, order_type)
        place_intents: list[tuple[str, str, str, float, str]] = []

        for cid in list(self.known_markets):
            # Skip markets where all ladder rungs have filled.
            if self._cid_fill_count.get(cid, 0) >= self.ladder_rungs:
                continue

            outcomes = self.market_outcomes.get(cid, [])
            if len(outcomes) < 2:
                continue

            # Record book snapshot for trend features (once per cid per tick).
            if self._model:
                self._record_book_snapshot(cid)

            for outcome in outcomes:
                token_id = self.market_tokens.get((cid, outcome))
                if not token_id:
                    continue

                bid, bid_sz, ask, ask_sz = self.polymarket.get_best_levels(cid, outcome)
                if bid is not None:
                    self._last_bids[(cid, outcome)] = bid
                    self._last_book_update = now
                existing_key = (cid, outcome)

                # BUY signal: bid in [target_bid, max_bid] + filters
                if self._model:
                    # ML model replaces manual move/timing filters
                    p_win = self._check_model(cid, outcome, bid) if bid is not None else None
                    order_type = self._last_order_type.pop((cid, outcome), "skip")
                    bid_in_range = (
                        bid is not None
                        and self.target_bid <= bid <= self.max_bid
                        and p_win is not None
                        and order_type != "skip"
                    )
                    if bid_in_range and p_win is not None:
                        self._last_p_win[(cid, outcome)] = p_win
                        # For taker orders, buy at the ask instead of bidding
                        if order_type == "taker" and ask is not None:
                            self._last_order_type[(cid, outcome)] = "taker"
                else:
                    # Manual filters fallback
                    bid_in_range = (
                        bid is not None
                        and self.target_bid <= bid <= self.max_bid
                        and self._check_min_entry_time(cid)
                        and self._check_max_entry_time(cid)
                        and self._check_min_move(cid, outcome, bid)
                        and self._check_max_move(cid, outcome)
                    )

                if self.ladder_rungs > 1 and bid is not None and bid >= self.target_bid:
                    # ---- Sequential ladder: place only the next rung ----
                    if not self._check_min_entry_time(cid):
                        continue
                    if not self._check_max_entry_time(cid):
                        continue
                    next_idx = self._cid_fill_count.get(cid, 0)
                    if next_idx < len(self.rung_prices):
                        rung_price = self.rung_prices[next_idx]
                        # Move filters scaled to rung price.
                        if not self._check_min_move(cid, outcome, rung_price):
                            continue
                        if not self._check_max_move(cid, outcome):
                            continue
                        # Skip if rung would cross the book (post-only rejection)
                        if ask is not None and rung_price >= ask:
                            continue
                        rung_key = (cid, outcome, int(round(rung_price * 100)))
                        if rung_key not in self._rung_placed:
                            place_intents.append((cid, outcome, token_id, rung_price, "maker"))
                else:
                    # ---- Single-order mode (original logic) ----
                    existing_oid = self._orders_by_cid_outcome.get(existing_key)
                    existing_order = self.active_orders.get(existing_oid) if existing_oid else None

                    if existing_order:
                        if not bid_in_range:
                            cancel_ids.append(existing_order.order_id)
                        elif existing_order.price != bid:
                            if self.paper_mode and bid < existing_order.price:
                                self._process_fill(existing_order, now)
                                oid = existing_order.order_id
                                self.active_orders.pop(oid, None)
                                if self._orders_by_cid_outcome.get(existing_key) == oid:
                                    del self._orders_by_cid_outcome[existing_key]
                                self._cancel_other_side(cid, outcome)
                            # else: bid moved within range — keep existing order.
                            # DO NOT cancel+replace: the CLOB fills faster than
                            # cancels propagate, causing runaway duplicate fills.
                        # else: order still at correct price, do nothing.
                    else:
                        # No existing order — place if in range
                        if bid_in_range and budget_left >= self.order_size_usd:
                            ot = self._last_order_type.get((cid, outcome), "maker")
                            if ot == "taker" and ask is not None:
                                place_intents.append((cid, outcome, token_id, ask, "taker"))
                            else:
                                place_intents.append((cid, outcome, token_id, bid, "maker"))

        # Execute cancels.
        for oid in cancel_ids:
            order = self.active_orders.get(oid)
            if not order:
                continue
            key = (order.condition_id, order.outcome)
            if not self.paper_mode and self.executor:
                # Track cancel in-flight so late fills are caught.
                order.cancelled_at = now
                self._pending_cancels[oid] = order
                try:
                    await self.executor.cancel_order(oid)
                    # Cancel confirmed — safe to clean up everywhere.
                    self._pending_cancels.pop(oid, None)
                    self.active_orders.pop(oid, None)
                    if self._orders_by_cid_outcome.get(key) == oid:
                        del self._orders_by_cid_outcome[key]
                    self._rung_placed.discard(
                        (order.condition_id, order.outcome, int(round(order.price * 100))))
                    self._db_fire(self._db_delete_order(oid))
                except Exception as exc:
                    # Cancel request failed — order likely still live.
                    # Keep in both active_orders AND _pending_cancels.
                    # _fill_listener will handle late fills.
                    logger.warning(
                        "td_cancel_request_failed",
                        order_id=oid[:16],
                        error=str(exc)[:80],
                    )
            else:
                # Paper mode: instant cancel.
                self.active_orders.pop(oid, None)
                if self._orders_by_cid_outcome.get(key) == oid:
                    del self._orders_by_cid_outcome[key]
                self._rung_placed.discard(
                    (order.condition_id, order.outcome, int(round(order.price * 100))))
                self._db_fire(self._db_delete_order(oid))

        # Execute placements.
        placed = 0
        placed_cids_this_tick: set[str] = set()  # prevent BUY Up + SELL Down on same cid
        # Collect condition_ids with pending cancels — don't place new orders there.
        pending_cancel_cids = {o.condition_id for o in self._pending_cancels.values()}
        for cid, outcome, token_id, price, order_type in place_intents:
            if self._cid_fill_count.get(cid, 0) >= self.ladder_rungs:
                continue
            # Don't place new orders while a cancel is in-flight for this market.
            if cid in pending_cancel_cids:
                continue

            if self.ladder_rungs > 1:
                # Ladder mode: allow multiple orders per cid, dedup by rung.
                rung_key = (cid, outcome, int(round(price * 100)))
                if rung_key in self._rung_placed:
                    continue
            else:
                # Single-order mode: existing dedup.
                if cid in self.positions:
                    continue
                if (cid, outcome) in self._orders_by_cid_outcome:
                    continue
                if cid in placed_cids_this_tick:
                    continue
                if any(o.condition_id == cid for o in self.active_orders.values()):
                    continue

            if budget_left < self.order_size_usd:
                continue

            # Fair value entry guard: don't buy if price >> Chainlink fair value.
            if not self._check_fair_value_entry(cid, outcome, price, now):
                continue

            order_id = await self._place_order(
                cid, outcome, token_id, price, now, order_type=order_type,
            )
            if order_id:
                placed += 1
                placed_cids_this_tick.add(cid)
                budget_left -= self.order_size_usd

        # Periodic status (every 30s wall-clock).
        if now - self._last_status_time >= 30.0:
            self._last_status_time = now
            exposure = sum(p.size_usd for p in self.positions.values())
            winrate = self.total_wins / self.total_fills * 100 if self.total_fills else 0

            # Snapshot of current best bids per outcome.
            price_parts: list[str] = []
            for cid in list(self.known_markets)[:4]:
                for outcome in self.market_outcomes.get(cid, []):
                    bid, _, ask, _ = self.polymarket.get_best_levels(cid, outcome)
                    bid_s = f"{bid:.2f}" if bid else "?"
                    ask_s = f"{ask:.2f}" if ask else "?"
                    price_parts.append(f"{outcome[:1]}:{bid_s}/{ask_s}")

            print(
                f"[{time.strftime('%H:%M:%S')}] "
                f"cb={'ON' if (self.guard and self.guard.circuit_broken) else 'off'} "
                f"mkts={len(self.known_markets)} "
                f"orders={len(self.active_orders)} "
                f"pos={len(self.positions)} "
                f"fills={self.total_fills} "
                f"{self.total_wins}W-{self.total_losses}L "
                f"sl_ovr={self.stoploss_overrides} "
                f"pnl=${self.realized_pnl:+.2f} "
                f"exp=${exposure:.0f} "
                f"prices=[{' '.join(price_parts)}]"
            )

    # ------------------------------------------------------------------
    # Order management
    # ------------------------------------------------------------------

    async def _place_order(
        self, cid: str, outcome: str, token_id: str, price: float, now: float,
        *, order_type: str = "maker",
    ) -> Optional[str]:
        """Place a BUY order — GTC maker or FOK taker depending on order_type."""
        if not self.manager:
            return None

        is_taker = order_type == "taker"

        # Model-based sizing: scale order_size by confidence
        size_usd = self.order_size_usd
        p_win = self._last_p_win.pop((cid, outcome), None)
        if p_win is not None and self._model:
            # Scale: 0.55 -> 0.2x, 0.75 -> 1.0x, 1.0 -> 2.0x (linear)
            confidence_scale = max(0.2, min(2.0, (p_win - 0.5) / 0.25))
            size_usd = round(self.order_size_usd * confidence_scale, 2)
            logger.debug("model_sizing", cid=cid[:12], outcome=outcome,
                         p_win=round(p_win, 3), scale=round(confidence_scale, 2),
                         size_usd=size_usd, order_type=order_type)

        # Pre-register a placeholder BEFORE the await so the fill listener
        # can match fills that arrive while we're waiting for the API response.
        temp_id = f"_placing_{cid}_{outcome}_{int(round(price*100))}"
        placeholder = PassiveOrder(
            order_id=temp_id, condition_id=cid, outcome=outcome,
            token_id=token_id, price=price, size_usd=size_usd,
            placed_at=now,
        )
        self.active_orders[temp_id] = placeholder
        self._orders_by_cid_outcome[(cid, outcome)] = temp_id
        rung_key = (cid, outcome, int(round(price * 100)))
        self._rung_placed.add(rung_key)

        slug = _first_event_slug(self.known_markets.get(cid, {}))
        reason = "td_taker_fok" if is_taker else "td_maker_passive"
        intent = TradeIntent(
            condition_id=cid, token_id=token_id, outcome=outcome,
            side="BUY", price=price, size_usd=size_usd,
            reason=reason, title=slug, timestamp=now,
        )
        try:
            pending = await self.manager.place(
                intent, order_type="FOK" if is_taker else "GTC",
            )
        except Exception as exc:
            # Placement failed — clean up placeholder.
            self.active_orders.pop(temp_id, None)
            if self._orders_by_cid_outcome.get((cid, outcome)) == temp_id:
                del self._orders_by_cid_outcome[(cid, outcome)]
            self._rung_placed.discard(rung_key)
            logger.warning("order_place_exception", error=str(exc)[:80])
            return None

        if not pending.order_id:
            self.active_orders.pop(temp_id, None)
            if self._orders_by_cid_outcome.get((cid, outcome)) == temp_id:
                del self._orders_by_cid_outcome[(cid, outcome)]
            self._rung_placed.discard(rung_key)
            return None

        # FOK taker orders fill immediately (or reject entirely).
        # In paper mode we simulate an instant fill at the ask.
        if is_taker:
            self.active_orders.pop(temp_id, None)
            if self._orders_by_cid_outcome.get((cid, outcome)) == temp_id:
                del self._orders_by_cid_outcome[(cid, outcome)]
            order = PassiveOrder(
                order_id=pending.order_id, condition_id=cid, outcome=outcome,
                token_id=token_id, price=price, size_usd=size_usd,
                placed_at=now,
            )
            self._process_fill(order, now)
            self._cancel_other_side(cid, outcome)
            self._db_fire(self._db_save_order(order))
            self._db_fire(self._db_mark_filled(
                pending.order_id, size_usd / price, now,
            ))
            logger.info("td_taker_fok_filled", order_id=pending.order_id[:16],
                         outcome=outcome, price=price)
            return pending.order_id

        # Fill listener may have already processed a fill for the placeholder.
        # For single-order mode: position means this rung was filled.
        # For ladder: position may exist from a previous rung — only skip if
        # THIS rung was filled (detected by temp_id no longer in active_orders).
        if temp_id not in self.active_orders:
            if self._orders_by_cid_outcome.get((cid, outcome)) == temp_id:
                del self._orders_by_cid_outcome[(cid, outcome)]
            self._position_order_ids[cid] = pending.order_id
            order = PassiveOrder(
                order_id=pending.order_id, condition_id=cid, outcome=outcome,
                token_id=token_id, price=price, size_usd=self.order_size_usd,
                placed_at=now,
            )
            self._db_fire(self._db_save_order(order))
            self._db_fire(self._db_mark_filled(pending.order_id, self.positions[cid].shares, now))
            logger.info("td_fill_caught_during_placement", order_id=pending.order_id[:16])
            return pending.order_id

        # No fill yet — replace placeholder with real order.
        self.active_orders.pop(temp_id, None)
        order = PassiveOrder(
            order_id=pending.order_id, condition_id=cid, outcome=outcome,
            token_id=token_id, price=price, size_usd=self.order_size_usd,
            placed_at=now,
        )
        self.active_orders[pending.order_id] = order
        self._orders_by_cid_outcome[(cid, outcome)] = pending.order_id
        self._db_fire(self._db_save_order(order))
        return pending.order_id

    # ------------------------------------------------------------------
    # Fill detection
    # ------------------------------------------------------------------

    # Paper fill tuning: how long at best bid before simulated fill.
    PAPER_FILL_TIMEOUT: float = 30.0
    # Maximum spread for time-based fill to trigger.
    PAPER_FILL_MAX_SPREAD: float = 0.03

    def _check_fills_paper(self, now: float) -> None:
        """Paper mode: simulate maker fills via three conditions.

        1. Ask crossed: ask <= our bid price.
        2. Bid-through: bid dropped below our price — our level was consumed.
        3. Time-at-bid: order at best bid with tight spread for PAPER_FILL_TIMEOUT.
        """
        filled_ids: list[str] = []
        for order_id, order in self.active_orders.items():
            bid, _, ask, _ = self.polymarket.get_best_levels(
                order.condition_id, order.outcome
            )

            # 1. Ask crossed down to our bid.
            if ask is not None and ask <= order.price:
                self._process_fill(order, now)
                filled_ids.append(order_id)
            # 2. Bid dropped below our price — our level got consumed.
            elif bid is not None and bid < order.price:
                self._process_fill(order, now)
                filled_ids.append(order_id)
            # 3. Sitting at best bid with tight spread long enough.
            elif (
                bid is not None
                and ask is not None
                and bid == order.price
                and (ask - bid) <= self.PAPER_FILL_MAX_SPREAD
                and (now - order.placed_at) >= self.PAPER_FILL_TIMEOUT
            ):
                self._process_fill(order, now)
                filled_ids.append(order_id)

        for oid in filled_ids:
            order = self.active_orders.pop(oid, None)
            if order:
                key = (order.condition_id, order.outcome)
                if self._orders_by_cid_outcome.get(key) == oid:
                    del self._orders_by_cid_outcome[key]
                # Cancel other side only on first fill for this cid.
                if self._cid_fill_count.get(order.condition_id, 0) <= 1:
                    self._cancel_other_side(order.condition_id, order.outcome)

    def _process_fill(self, order: PassiveOrder, now: float) -> None:
        """Record a filled order as an open position (with scale-in for ladder)."""
        new_shares = order.size_usd / order.price

        existing = self.positions.get(order.condition_id)
        if existing and self.ladder_rungs > 1:
            # Scale-in: accumulate into existing position.
            old_shares = existing.shares
            total_shares = old_shares + new_shares
            avg_price = (old_shares * existing.entry_price + new_shares * order.price) / total_shares
            existing.shares = total_shares
            existing.entry_price = avg_price
            existing.size_usd += order.size_usd
        else:
            pos = OpenPosition(
                condition_id=order.condition_id,
                outcome=order.outcome,
                token_id=order.token_id,
                entry_price=order.price,
                size_usd=order.size_usd,
                shares=new_shares,
                filled_at=now,
            )
            self.positions[order.condition_id] = pos

        self._position_order_ids[order.condition_id] = order.order_id
        self._cid_fill_count[order.condition_id] = self._cid_fill_count.get(order.condition_id, 0) + 1
        # Initialize bid_max tracking for stop-loss.
        if order.condition_id not in self._position_bid_max:
            self._position_bid_max[order.condition_id] = order.price
        self.total_fills += 1
        self._db_fire(self._db_mark_filled(order.order_id, new_shares, now))

        # Underlying move at fill time (reuses shared helper).
        dir_move = self._get_dir_move(order.condition_id, order.outcome)
        ref = self._ref_prices.get(order.condition_id)
        sym = self._cid_chainlink_symbol.get(order.condition_id)
        current = self.chainlink_feed.get_price(sym) if sym and self.chainlink_feed else None

        logger.info(
            "td_order_filled",
            condition_id=order.condition_id[:16],
            outcome=order.outcome,
            price=order.price,
            shares=round(new_shares, 2),
            total_fills=self.total_fills,
            paper=self.paper_mode,
            ref_price=ref,
            chainlink_price=current,
            dir_move_pct=round(dir_move, 3) if dir_move is not None else None,
        )

        # Persist fill to DB + Telegram via TradeManager.
        if self.manager:
            self.manager._pending.pop(order.order_id, None)
            slug = _first_event_slug(self.known_markets.get(order.condition_id, {}))
            intent = TradeIntent(
                condition_id=order.condition_id,
                token_id=order.token_id,
                outcome=order.outcome,
                side="BUY",
                price=order.price,
                size_usd=order.size_usd,
                reason="td_maker_passive",
                title=slug,
                timestamp=now,
            )
            fill_result = FillResult(
                filled=True,
                shares=new_shares,
                avg_price=order.price,
            )
            analytics = {
                "dir_move_pct": round(dir_move, 3) if dir_move is not None else None,
                "minutes_into_slot": round((now - self._cid_slot_ts[order.condition_id]) / 60, 1)
                    if order.condition_id in self._cid_slot_ts else None,
            }
            self._cid_fill_analytics[order.condition_id] = analytics
            move_str = f"{dir_move:+.2f}%" if dir_move is not None else None
            timing = analytics.get("minutes_into_slot")
            timing_str = f"{timing:.0f}m" if timing is not None else None
            context_parts = [p for p in [
                f"move {move_str}" if move_str else None,
                f"entry {timing_str}" if timing_str else None,
            ] if p]
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self.manager.record_fill_direct(
                    intent, fill_result,
                    execution_mode="paper_fill" if self.paper_mode else "live_fill",
                    extra_state={"condition_id": order.condition_id, **analytics},
                    notify_context=" | ".join(context_parts) if context_parts else None,
                ))
            except RuntimeError:
                pass

    def _cancel_other_side(self, cid: str, filled_outcome: str) -> None:
        """Cancel ALL orders on the opposite outcome (supports ladder)."""
        outcomes = self.market_outcomes.get(cid, [])
        for outcome in outcomes:
            if outcome == filled_outcome:
                continue
            # Cancel all active orders on this (cid, outcome).
            self._orders_by_cid_outcome.pop((cid, outcome), None)
            oids_to_cancel = [oid for oid, o in self.active_orders.items()
                              if o.condition_id == cid and o.outcome == outcome]
            for oid in oids_to_cancel:
                order = self.active_orders.pop(oid)
                # Clean up rung tracking.
                self._rung_placed.discard((cid, outcome, int(round(order.price * 100))))
                if not self.paper_mode and self.executor:
                    order.cancelled_at = time.time()
                    self._pending_cancels[oid] = order
                    asyncio.create_task(self._async_cancel(oid))
                else:
                    self._db_fire(self._db_delete_order(oid))
                logger.debug("td_other_side_cancelled", outcome=outcome)

    async def _async_cancel(self, order_id: str) -> None:
        try:
            await self.executor.cancel_order(order_id)
            # Don't pop from _pending_cancels immediately — the CLOB may
            # return success even for orders that were already filled.
            # The stale-cancel expiry (30s) will clean up; meanwhile the
            # fill listener can still match late WS events.
            logger.debug("td_cancel_api_ok", order_id=order_id[:16])
        except Exception:
            # Keep in _pending_cancels; _fill_listener or expiry will handle it.
            pass

    async def _cancel_all_orders(self, reason: str) -> None:
        """Cancel every active order (stale escalation / emergency)."""
        count = len(self.active_orders)
        for oid in list(self.active_orders):
            order = self.active_orders.pop(oid)
            key = (order.condition_id, order.outcome)
            if self._orders_by_cid_outcome.get(key) == oid:
                del self._orders_by_cid_outcome[key]
            self._rung_placed.discard(
                (order.condition_id, order.outcome, int(round(order.price * 100)))
            )
            if not self.paper_mode and self.executor:
                order.cancelled_at = time.time()
                self._pending_cancels[oid] = order
                asyncio.create_task(self._async_cancel(oid))
            else:
                self._db_fire(self._db_delete_order(oid))
        logger.warning("td_all_orders_cancelled", reason=reason, count=count)

    async def _fill_listener(self) -> None:
        """Drain fills from WS User channel (live mode real-time detection)."""
        if not self.user_feed:
            return
        while True:
            try:
                evt: UserTradeEvent = await self.user_feed.fills.get()
            except asyncio.CancelledError:
                break

            # Skip if all ladder rungs already filled for this market.
            if self._cid_fill_count.get(evt.market, 0) >= self.ladder_rungs:
                continue
            # For single-order mode, also skip if position already exists.
            if self.ladder_rungs <= 1 and evt.market in self.positions:
                continue

            matched_order: Optional[PassiveOrder] = None
            matched_id: Optional[str] = None
            source = "active"

            # --- Priority 1: exact maker_order_id match (prevents phantom fills
            # from partial fills being mis-attributed to the next rung order) ---
            if evt.maker_order_id:
                if evt.maker_order_id in self.active_orders:
                    matched_id = evt.maker_order_id
                    matched_order = self.active_orders[matched_id]
                elif evt.maker_order_id in self._pending_cancels:
                    matched_id = evt.maker_order_id
                    matched_order = self._pending_cancels[matched_id]
                    source = "pending_cancel"
                    logger.warning(
                        "td_late_fill_from_cancelled_order",
                        order_id=matched_id[:16],
                        condition_id=matched_order.condition_id[:16],
                        outcome=matched_order.outcome,
                        price=matched_order.price,
                    )

            # --- Priority 2: condition_id match for PLACEHOLDER orders only ---
            # Placeholders (id starts with "_placing_") exist while the API call
            # is in-flight; we don't yet know the real order_id, so fall back to
            # condition_id matching restricted to placeholders.
            if not matched_order:
                for oid, order in self.active_orders.items():
                    if oid.startswith("_placing_") and order.condition_id == evt.market:
                        matched_order = order
                        matched_id = oid
                        break

            # --- Priority 3: condition_id match in pending_cancels (placeholders) ---
            if not matched_order:
                for oid, order in self._pending_cancels.items():
                    if oid.startswith("_placing_") and order.condition_id == evt.market:
                        matched_order = order
                        matched_id = oid
                        source = "pending_cancel"
                        logger.warning(
                            "td_late_fill_from_cancelled_order",
                            order_id=oid[:16],
                            condition_id=order.condition_id[:16],
                            outcome=order.outcome,
                            price=order.price,
                        )
                        break

            # --- Fallback: broad condition_id match (only when maker_order_id
            # is missing, e.g. taker fills or WS edge cases) ---
            if not matched_order and not evt.maker_order_id:
                for oid, order in self.active_orders.items():
                    if order.condition_id == evt.market:
                        matched_order = order
                        matched_id = oid
                        break
                if not matched_order:
                    for oid, order in self._pending_cancels.items():
                        if order.condition_id == evt.market:
                            matched_order = order
                            matched_id = oid
                            source = "pending_cancel"
                            logger.warning(
                                "td_late_fill_from_cancelled_order",
                                order_id=oid[:16],
                                condition_id=order.condition_id[:16],
                                outcome=order.outcome,
                                price=order.price,
                            )
                            break

            if not matched_order or not matched_id:
                logger.warning(
                    "td_fill_unmatched",
                    market=evt.market[:16],
                    asset_id=evt.asset_id[:16],
                    side=evt.side,
                    price=evt.price,
                    size=evt.size,
                    status=evt.status,
                )
                # Even if unmatched, count it to prevent re-ordering.
                self._cid_fill_count[evt.market] = self._cid_fill_count.get(evt.market, 0) + 1
                continue

            now = time.time()
            self._process_fill(matched_order, now)
            if source == "active":
                del self.active_orders[matched_id]
            else:
                self._pending_cancels.pop(matched_id, None)
            key = (matched_order.condition_id, matched_order.outcome)
            if self._orders_by_cid_outcome.get(key) == matched_id:
                del self._orders_by_cid_outcome[key]
            # Don't cancel other side on every ladder fill — only on first fill.
            if self._cid_fill_count.get(matched_order.condition_id, 0) <= 1:
                self._cancel_other_side(matched_order.condition_id, matched_order.outcome)

    # ------------------------------------------------------------------
    # Settlement
    # ------------------------------------------------------------------

    async def _prune_expired(self) -> None:
        """Settle positions and clean up expired markets."""
        now = time.time()
        to_remove: list[str] = []

        for cid, mkt in self.known_markets.items():
            slug = _first_event_slug(mkt).lower()
            parts = slug.rsplit("-", 1)
            if len(parts) == 2 and parts[1].isdigit():
                slot_end = int(parts[1]) + 900  # slot_ts + 15 min
                if now > slot_end + 300:  # 5 min grace for resolution
                    to_remove.append(cid)

        actually_removed: list[str] = []
        for cid in to_remove:
            # Cancel ALL remaining orders for this market (supports ladder).
            oids_to_cancel = [oid for oid, o in self.active_orders.items()
                              if o.condition_id == cid]
            for oid in oids_to_cancel:
                order = self.active_orders.pop(oid)
                self._rung_placed.discard(
                    (cid, order.outcome, int(round(order.price * 100))))
                self._db_fire(self._db_delete_order(oid))
                if not self.paper_mode and self.executor:
                    try:
                        await self.executor.cancel_order(oid)
                    except Exception:
                        pass
            for outcome in self.market_outcomes.get(cid, []):
                self._orders_by_cid_outcome.pop((cid, outcome), None)

            # Settle position if held.
            pos = self.positions.pop(cid, None)
            if pos:
                settled = await self._settle_position(pos, now)
                if not settled:
                    # Deferred — keep market data for retry on next prune cycle
                    continue

            actually_removed.append(cid)

            # Clean up fill count.
            self._cid_fill_count.pop(cid, None)

            # Unsubscribe.
            try:
                await self.polymarket.unsubscribe_market(cid)
            except Exception:
                pass

            self.known_markets.pop(cid, None)
            self._ref_prices.pop(cid, None)
            self._cid_chainlink_symbol.pop(cid, None)
            self._cid_slot_ts.pop(cid, None)
            self._cid_fill_analytics.pop(cid, None)
            self._book_history.pop(cid, None)
            self._position_bid_max.pop(cid, None)
            self._position_last_bid.pop(cid, None)
            for outcome in self.market_outcomes.pop(cid, []):
                self._last_bids.pop((cid, outcome), None)

        if actually_removed:
            logger.info("td_markets_pruned", count=len(actually_removed), remaining=len(self.known_markets))

        # Settle orphaned positions whose markets are no longer known
        # (e.g. loaded from DB after restart, but the 15-min slot expired).
        orphan_cids = [cid for cid in self.positions if cid not in self.known_markets]
        for cid in orphan_cids:
            pos = self.positions.pop(cid)
            logger.warning("td_orphan_position_settling", condition_id=cid[:16], outcome=pos.outcome)
            await self._settle_position(pos, now)

    async def _query_resolution(self, pos: OpenPosition) -> Optional[bool]:
        """Query Gamma/CLOB API for actual market resolution.

        Returns True if the token resolved to 1, False if 0, None if unknown.
        Tries Gamma event slug first, then CLOB condition_id as fallback.
        """
        if not self._http_client:
            return None

        # --- Attempt 1: Gamma API via event slug ---
        GAMMA_URL = "https://gamma-api.polymarket.com"
        slug = _first_event_slug(self.known_markets.get(pos.condition_id, {}))

        if slug:
            try:
                resp = await self._http_client.get(
                    f"{GAMMA_URL}/events",
                    params={"slug": slug},
                    timeout=10.0,
                )
                if resp.status_code == 200:
                    events = resp.json()
                    if events:
                        mkt = events[0].get("markets", [{}])[0]
                        outcome_prices_raw = json.loads(mkt.get("outcomePrices", "[]"))
                        outcomes_raw = json.loads(mkt.get("outcomes", "[]"))

                        for i, outcome in enumerate(outcomes_raw):
                            if outcome == pos.outcome and i < len(outcome_prices_raw):
                                price = float(outcome_prices_raw[i])
                                if price >= 0.9:
                                    return True
                                if price <= 0.1:
                                    return False
            except Exception as exc:
                logger.warning(
                    "gamma_resolution_failed",
                    condition_id=pos.condition_id[:16],
                    slug=slug,
                    error=str(exc)[:60],
                )

        # --- Attempt 2: CLOB API via condition_id (works for orphans too) ---
        CLOB_URL = settings.POLYMARKET_CLOB_HTTP
        try:
            resp = await self._http_client.get(
                f"{CLOB_URL}/markets/{pos.condition_id}",
                timeout=10.0,
            )
            if resp.status_code == 200:
                mkt_data = resp.json()
                # CLOB returns tokens array with outcome + price
                tokens = mkt_data.get("tokens", [])
                for tok in tokens:
                    if tok.get("outcome") == pos.outcome:
                        price = float(tok.get("price", 0.5))
                        if price >= 0.9:
                            return True
                        if price <= 0.1:
                            return False
        except Exception as exc:
            logger.warning(
                "clob_resolution_failed",
                condition_id=pos.condition_id[:16],
                error=str(exc)[:60],
            )

        return None

    async def _settle_position(self, pos: OpenPosition, now: float, *, allow_defer: bool = True) -> bool:
        """Determine win/loss from Gamma/CLOB API resolution.

        Won if token resolves to 1, pnl = shares * (1 - entry).
        Returns True if settled, False if deferred (resolution unknown).
        """
        # Primary: query APIs for actual resolved outcome prices
        resolution = await self._query_resolution(pos)

        if resolution is not None:
            token_resolved_1 = resolution
        else:
            # Fallback: last book state — only trust clear signals (bid >= 0.9 or <= 0.1)
            bid, _, _, _ = self.polymarket.get_best_levels(pos.condition_id, pos.outcome)
            if bid is None:
                bid = self._last_bids.get((pos.condition_id, pos.outcome))

            if bid is not None and bid >= 0.9:
                token_resolved_1 = True
            elif bid is not None and bid <= 0.1:
                token_resolved_1 = False
            else:
                # Cannot determine resolution — defer settlement
                age_min = (now - pos.filled_at) / 60
                if allow_defer and age_min < 60:
                    logger.warning(
                        "td_settle_deferred",
                        condition_id=pos.condition_id[:16],
                        outcome=pos.outcome,
                        bid=bid,
                        age_min=round(age_min, 1),
                    )
                    # Put position back so next prune cycle retries
                    self.positions[pos.condition_id] = pos
                    return False
                # Too old to defer — force settle using bid if available, else loss
                token_resolved_1 = bid is not None and bid >= 0.5
                logger.error(
                    "td_settle_forced_unknown",
                    condition_id=pos.condition_id[:16],
                    outcome=pos.outcome,
                    bid=bid,
                    resolved_1=token_resolved_1,
                    age_min=round(age_min, 1),
                )

        won = token_resolved_1
        pnl = pos.shares * (1.0 - pos.entry_price) if won else -pos.size_usd
        self._position_bid_max.pop(pos.condition_id, None)
        self._position_last_bid.pop(pos.condition_id, None)

        if won:
            self.total_wins += 1
        else:
            self.total_losses += 1
        self.realized_pnl += pnl

        # Mark settled in DB.
        oid = self._position_order_ids.pop(pos.condition_id, None)
        if oid:
            self._db_fire(self._db_mark_settled(oid, pnl, now))

        if self.guard:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self.guard.record_result(pnl=pnl, won=won))
            except RuntimeError:
                pass

        logger.info(
            "td_position_settled",
            condition_id=pos.condition_id[:16],
            outcome=pos.outcome,
            entry_price=pos.entry_price,
            won=won,
            pnl=round(pnl, 4),
            total_pnl=round(self.realized_pnl, 4),
            record=f"{self.total_wins}W-{self.total_losses}L",
        )

        if self.manager:
            slug = _first_event_slug(self.known_markets.get(pos.condition_id, {}))
            resolution_price = 1.0 if token_resolved_1 else 0.0
            settle_intent = TradeIntent(
                condition_id=pos.condition_id,
                token_id=pos.token_id,
                outcome=pos.outcome,
                side="SELL",
                price=pos.entry_price,
                size_usd=pos.size_usd,
                reason="settlement",
                title=slug,
                timestamp=now,
            )
            settle_fill = FillResult(
                filled=True,
                shares=pos.shares,
                avg_price=resolution_price,
                pnl_delta=pnl,
            )
            try:
                analytics = self._cid_fill_analytics.pop(pos.condition_id, None)
                # Build context for Telegram notification
                context_parts = []
                if analytics:
                    dir_move = analytics.get("dir_move_pct")
                    timing = analytics.get("minutes_into_slot")
                    if dir_move is not None:
                        context_parts.append(f"move {dir_move:+.2f}%")
                    if timing is not None:
                        context_parts.append(f"entry {timing:.0f}m")
                notify_ctx = " | ".join(context_parts) if context_parts else None

                loop = asyncio.get_running_loop()
                loop.create_task(self.manager.record_settle_direct(
                    settle_intent, settle_fill,
                    extra_state=analytics,
                    notify_context=notify_ctx,
                ))
            except RuntimeError:
                pass

        return True

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    async def run(self) -> None:
        await self._load_db_state()
        await self.polymarket.connect()

        if self.chainlink_feed:
            await self.chainlink_feed.connect()
            # Give WS a moment to receive initial prices.
            await asyncio.sleep(2)

        if self.user_feed:
            try:
                await self.user_feed.connect()
                logger.info("user_ws_connected")
            except Exception as exc:
                logger.warning("user_ws_connect_failed", error=str(exc))
                self.user_feed = None

        timeout = httpx.Timeout(20.0, connect=10.0)
        limits = httpx.Limits(max_connections=20, max_keepalive_connections=10)
        async with httpx.AsyncClient(timeout=timeout, limits=limits, http2=True) as client:
            self._http_client = client
            await self._discover_markets()

            logger.info(
                "crypto_td_maker_started",
                symbols=self.symbols,
                paper=self.paper_mode,
                buy_range=f"[{self.target_bid}, {self.max_bid}]",
                ladder_rungs=self.ladder_rungs,
                rung_prices=self.rung_prices,
                order_size=self.order_size_usd,
                max_exposure=self.max_total_exposure_usd,
                markets=len(self.known_markets),
                user_ws=self.user_feed is not None,
                chainlink_ws=self.chainlink_feed is not None and self.chainlink_feed.is_connected,
            )

            tasks: list[Any] = [
                self.market_discovery_loop(),
                self.maker_loop(),
            ]
            if self.user_feed:
                tasks.append(self._fill_listener())

            try:
                await asyncio.gather(*tasks)
            finally:
                self._http_client = None
                if self.chainlink_feed:
                    await self.chainlink_feed.disconnect()
                if self.user_feed:
                    await self.user_feed.disconnect()
                await self.polymarket.disconnect()
                if self.manager:
                    await self.manager.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Passive time-decay maker for crypto 15-min markets"
    )
    p.add_argument("--symbols", type=str, default="BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT")
    p.add_argument("--paper", action="store_true", default=True)
    p.add_argument("--live", action="store_true", default=False)
    p.add_argument(
        "--target-bid", type=float, default=0.75,
        help="Min bid price to enter (default: 0.75)",
    )
    p.add_argument(
        "--max-bid", type=float, default=0.85,
        help="Max bid price to enter (default: 0.85)",
    )
    p.add_argument("--wallet", type=float, default=0.0,
                    help="Wallet USD (0 = auto-detect from Polymarket balance)")
    p.add_argument("--order-size", type=float, default=0.0,
                    help="USD per order (0 = derive from wallet * 0.025)")
    p.add_argument("--max-exposure", type=float, default=0.0,
                    help="Max total USD exposure (0 = derive from wallet * 0.50)")
    p.add_argument("--discovery-interval", type=float, default=60.0)
    p.add_argument("--maker-interval", type=float, default=0.5, help="Maker loop tick interval")
    p.add_argument(
        "--ladder-rungs", type=int, default=1,
        help="Number of price ladder rungs (1=single order, >1=ladder with scale-in)",
    )
    p.add_argument("--strategy-tag", type=str, default="crypto_td_maker")
    p.add_argument("--db-url", type=str, default=settings.DATABASE_URL)
    p.add_argument("--cb-max-losses", type=int, default=5, help="Circuit breaker: max consecutive losses")
    p.add_argument("--cb-max-drawdown", type=float, default=-50.0, help="Circuit breaker: max session drawdown USD")
    p.add_argument("--cb-stale-seconds", type=float, default=30.0, help="Circuit breaker: book staleness threshold")
    p.add_argument("--cb-stale-cancel", type=float, default=120.0, help="Stale escalation: cancel all orders after N seconds")
    p.add_argument("--cb-stale-exit", type=float, default=300.0, help="Stale escalation: exit process after N seconds")
    p.add_argument("--cb-daily-limit", type=float, default=-200.0, help="Global daily loss limit USD")
    p.add_argument(
        "--min-move-pct", type=float, default=0.0,
        help="Min underlying price move (%%) in bet direction to enter (0=disabled)",
    )
    p.add_argument(
        "--max-move-pct", type=float, default=0.0,
        help="Max underlying price move (%%) — reject excessive volatility (0=disabled)",
    )
    p.add_argument(
        "--min-entry-minutes", type=float, default=0.0,
        help="Min minutes into slot before placing orders (0=disabled)",
    )
    p.add_argument(
        "--max-entry-minutes", type=float, default=0.0,
        help="Max minutes into slot — stop entering after this (0=disabled)",
    )
    p.add_argument(
        "--model-path", type=str, default="",
        help="Path to trained XGBoost model (.joblib). Model replaces manual filters.",
    )
    p.add_argument(
        "--hybrid-skip-below", type=float, default=0.55,
        help="ML p_win below this → skip (default: 0.55)",
    )
    p.add_argument(
        "--hybrid-taker-above", type=float, default=0.72,
        help="ML p_win above this → taker FOK at ask (default: 0.72)",
    )
    p.add_argument(
        "--stoploss-peak", type=float, default=0.0,
        help="Stop-loss: min bid_max to arm (0=disabled, recommended: 0.75)",
    )
    p.add_argument(
        "--stoploss-exit", type=float, default=0.0,
        help="Stop-loss: sell when bid drops to this after peak (0=disabled)",
    )
    p.add_argument(
        "--stoploss-fair-margin", type=float, default=0.10,
        help="Override stop-loss when Chainlink fair value > exit + margin (default: 0.10)",
    )
    p.add_argument(
        "--entry-fair-margin", type=float, default=0.0,
        help="Block entry when price > fair value + margin (0=disabled)",
    )
    p.add_argument(
        "--exit-model-path", type=str, default="",
        help="Path to trained exit model (.joblib). Replaces rule-based stop-loss.",
    )
    p.add_argument(
        "--exit-threshold", type=float, default=0.35,
        help="ML exit: sell when P(win) drops below this (default: 0.35)",
    )
    return p


async def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    symbols = [
        s if s.endswith("USDT") else s + "USDT"
        for s in (tok.strip().upper() for tok in args.symbols.split(","))
        if s
    ]
    paper_mode = not args.live
    strategy_tag = args.strategy_tag.strip() or "crypto_td_maker"

    run_id = f"{strategy_tag}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

    executor: Optional[PolymarketExecutor] = None
    if not paper_mode:
        executor = PolymarketExecutor.from_settings()

    manager = TradeManager(
        executor=executor,
        strategy="CryptoTDMaker",
        paper=paper_mode,
        db_url=args.db_url,
        event_type=TD_MAKER_EVENT_TYPE,
        run_id=run_id,
        notify_bids=False,
        notify_fills=not paper_mode,
        notify_closes=not paper_mode,
    )

    guard = RiskGuard(
        strategy_tag=strategy_tag,
        db_url=args.db_url,
        max_consecutive_losses=args.cb_max_losses,
        max_drawdown_usd=args.cb_max_drawdown,
        stale_seconds=args.cb_stale_seconds,
        stale_cancel_seconds=args.cb_stale_cancel,
        stale_exit_seconds=args.cb_stale_exit,
        daily_loss_limit_usd=args.cb_daily_limit,
        telegram_alerter=manager._alerter,
    )
    await guard.initialize()

    polymarket = PolymarketFeed()

    user_feed: Optional[PolymarketUserFeed] = None
    if not paper_mode and settings.POLYMARKET_API_KEY:
        api_key = settings.POLYMARKET_API_KEY
        api_secret = settings.POLYMARKET_API_SECRET
        api_passphrase = settings.POLYMARKET_API_PASSPHRASE
        if api_key and api_secret and api_passphrase:
            user_feed = PolymarketUserFeed(
                api_key=api_key,
                api_secret=api_secret,
                api_passphrase=api_passphrase,
            )

    # Resolve wallet → sizing
    wallet = args.wallet
    if wallet <= 0 and not paper_mode:
        wallet = await manager.get_wallet_balance()
        if wallet <= 0:
            logger.error("auto_wallet_failed", hint="pass --wallet explicitly")
            return
        logger.info("auto_wallet_detected", wallet_usd=round(wallet, 2))

    order_size = args.order_size if args.order_size > 0 else max(wallet * 0.025, 1.0)
    max_exposure = args.max_exposure if args.max_exposure > 0 else max(wallet * 0.50, 50.0)

    chainlink_feed = ChainlinkFeed()

    maker = CryptoTDMaker(
        executor=executor,
        polymarket=polymarket,
        user_feed=user_feed,
        manager=manager,
        symbols=symbols,
        target_bid=args.target_bid,
        max_bid=args.max_bid,
        order_size_usd=order_size,
        max_total_exposure_usd=max_exposure,
        paper_mode=paper_mode,
        discovery_interval=args.discovery_interval,
        maker_loop_interval=args.maker_interval,
        strategy_tag=strategy_tag,
        guard=guard,
        db_url=args.db_url,
        ladder_rungs=args.ladder_rungs,
        min_move_pct=args.min_move_pct,
        max_move_pct=args.max_move_pct,
        min_entry_minutes=args.min_entry_minutes,
        max_entry_minutes=args.max_entry_minutes,
        chainlink_feed=chainlink_feed,
        model_path=args.model_path,
        hybrid_skip_below=args.hybrid_skip_below,
        hybrid_taker_above=args.hybrid_taker_above,
        stoploss_peak=args.stoploss_peak,
        stoploss_exit=args.stoploss_exit,
        stoploss_fair_margin=args.stoploss_fair_margin,
        entry_fair_margin=args.entry_fair_margin,
        exit_model_path=args.exit_model_path,
        exit_threshold=args.exit_threshold,
    )

    wallet_src = "auto" if args.wallet <= 0 else "manual"
    rung_str = ", ".join(f"{p:.2f}" for p in maker.rung_prices)
    print(f"=== Crypto TD Maker {'(PAPER)' if paper_mode else '(LIVE)'} ===")
    print(f"  Symbols:     {', '.join(symbols)}")
    print(f"  Wallet:      ${wallet:.0f} ({wallet_src})")
    print(f"  BUY range:   [{args.target_bid}, {args.max_bid}]")
    if args.ladder_rungs > 1:
        print(f"  Ladder:      {args.ladder_rungs} rungs at [{rung_str}]")
    print(f"  Order size:  ${order_size:.2f}/rung")
    print(f"  Max exposure: ${max_exposure:.2f}")
    if args.min_entry_minutes > 0 or args.max_entry_minutes > 0:
        parts = []
        if args.min_entry_minutes > 0:
            parts.append(f">={args.min_entry_minutes}m")
        if args.max_entry_minutes > 0:
            parts.append(f"<={args.max_entry_minutes}m")
        print(f"  Entry time:  {' & '.join(parts)} into slot")
    if args.min_move_pct > 0 or args.max_move_pct > 0:
        parts = []
        if args.min_move_pct > 0:
            parts.append(f">={args.min_move_pct}%")
        if args.max_move_pct > 0:
            parts.append(f"<={args.max_move_pct}%")
        print(f"  Move filter: {' & '.join(parts)} (Chainlink directional)")
    if args.model_path:
        print(f"  ML model:    {args.model_path}")
        print(f"  Hybrid:      skip<{args.hybrid_skip_below} | "
              f"maker {args.hybrid_skip_below}-{args.hybrid_taker_above} | "
              f"taker>{args.hybrid_taker_above}")
    if args.stoploss_peak > 0:
        print(f"  Stop-loss:   peak>={args.stoploss_peak} & bid<={args.stoploss_exit} → sell FOK"
              f" (fair override margin={args.stoploss_fair_margin})")
    if args.exit_model_path:
        print(f"  Exit model:  {args.exit_model_path}")
        print(f"  Exit thresh: P(win)<{args.exit_threshold} → sell FOK")
    print(f"  Strategy:    Passive maker on 15-min crypto markets")
    print()

    await maker.run()


if __name__ == "__main__":
    if uvloop is not None:
        uvloop.install()
    asyncio.run(main())
