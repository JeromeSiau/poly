#!/usr/bin/env python3
"""Two-sided passive maker for Polymarket 15-min crypto markets.

STRATEGY (inspired by 0x8dxd):
    On Polymarket 15-min crypto binary markets (BTC/ETH up/down), post
    passive GTC bids on BOTH the Up and Down outcomes simultaneously.
    When combined bid prices < 1.0, any combination of fills is profitable
    (maker fees = 0%).

    How it works:
      1. Subscribe to orderbooks for all active 15-min crypto markets.
      2. On each tick, check both outcomes. When the best BIDs satisfy
         bid_up + bid_down < 1.0 - min_edge and each bid is in
         [min_bid, max_bid], place GTC buys at both bid prices.
      3. If BOTH fill: guaranteed profit = 1.0 - (cost_up + cost_down).
      4. If ONE fills: directional bet held to resolution (like TD maker).
      5. Near expiry, cancel unfilled orders to limit one-sided exposure.

    This is a structural arbitrage via maker execution, separate from
    the directional CryptoTDMaker and the taker-based CryptoTwoSided.

USAGE:
    ./run scripts/run_crypto_two_sided_maker.py --paper     # default
    ./run scripts/run_crypto_two_sided_maker.py --live      # real orders
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
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
from src.utils.crypto_markets import fetch_crypto_markets
from src.utils.parsing import parse_json_list, _first_event_slug
from src.execution import TradeManager, TradeIntent, FillResult
from src.risk.guard import RiskGuard

logger = structlog.get_logger()

TWO_SIDED_MAKER_EVENT_TYPE = "crypto_2s_maker"


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
class TwoSidedPair:
    """Tracks both sides of a two-sided maker position on one market."""
    condition_id: str
    up_token_id: str
    down_token_id: str
    # Fill state
    up_filled: bool = False
    down_filled: bool = False
    up_price: float = 0.0
    down_price: float = 0.0
    up_shares: float = 0.0
    down_shares: float = 0.0
    up_size_usd: float = 0.0
    down_size_usd: float = 0.0
    up_order_id: str = ""
    down_order_id: str = ""
    up_filled_at: float = 0.0
    down_filled_at: float = 0.0

    @property
    def both_filled(self) -> bool:
        return self.up_filled and self.down_filled

    @property
    def any_filled(self) -> bool:
        return self.up_filled or self.down_filled

    @property
    def combined_cost(self) -> float:
        return self.up_price + self.down_price

    @property
    def structural_edge(self) -> float:
        if self.both_filled:
            return 1.0 - self.combined_cost
        return 0.0


# ---------------------------------------------------------------------------
# CryptoTwoSidedMaker
# ---------------------------------------------------------------------------

class CryptoTwoSidedMaker:
    """Two-sided passive maker for 15-min crypto binary markets.

    Posts GTC bids on BOTH Up and Down outcomes when the combined cost
    is below 1.0 - min_edge. Holds filled positions to resolution.
    """

    def __init__(
        self,
        *,
        executor: Optional[PolymarketExecutor],
        polymarket: PolymarketFeed,
        user_feed: Optional[PolymarketUserFeed],
        manager: Optional[TradeManager] = None,
        symbols: list[str],
        min_edge: float = 0.02,
        max_bid: float = 0.60,
        min_bid: float = 0.30,
        order_size_usd: float = 10.0,
        max_total_exposure_usd: float = 200.0,
        max_concurrent_pairs: int = 10,
        cancel_before_close: int = 180,
        paper_mode: bool = True,
        discovery_interval: float = 60.0,
        maker_loop_interval: float = 0.5,
        strategy_tag: str = "crypto_2s_maker",
        guard: Optional[RiskGuard] = None,
        db_url: str = "",
    ) -> None:
        self.executor = executor
        self.polymarket = polymarket
        self.user_feed = user_feed
        self.manager = manager
        self.symbols = symbols
        self.min_edge = min_edge
        self.max_bid = max_bid
        self.min_bid = min_bid
        self.order_size_usd = order_size_usd
        self.max_total_exposure_usd = max_total_exposure_usd
        self.max_concurrent_pairs = max_concurrent_pairs
        self.cancel_before_close = cancel_before_close
        self.paper_mode = paper_mode
        self.discovery_interval = discovery_interval
        self.maker_loop_interval = maker_loop_interval
        self.strategy_tag = strategy_tag
        self.guard = guard
        self._db_url = db_url
        self._last_book_update: float = time.time()

        # State
        self.known_markets: dict[str, dict[str, Any]] = {}
        self.market_tokens: dict[tuple[str, str], str] = {}  # (cid, outcome) -> token_id
        self.market_outcomes: dict[str, list[str]] = {}  # cid -> ["Up", "Down"]
        self.active_orders: dict[str, PassiveOrder] = {}  # order_id -> order
        self._orders_by_cid_outcome: dict[tuple[str, str], str] = {}
        self.pairs: dict[str, TwoSidedPair] = {}  # cid -> pair
        self._pending_cancels: dict[str, PassiveOrder] = {}
        self._http_client: Optional[httpx.AsyncClient] = None
        self._cycle_count: int = 0
        self._last_status_time: float = 0.0
        self._last_bids: dict[tuple[str, str], float] = {}
        # Map (cid, outcome) -> order_id for DB settlement tracking.
        self._position_order_ids: dict[tuple[str, str], str] = {}

        # Stats
        self.total_fills: int = 0
        self.total_both_fills: int = 0
        self.total_wins: int = 0
        self.total_losses: int = 0
        self.realized_pnl: float = 0.0

    # ------------------------------------------------------------------
    # DB persistence (reuses TDMakerOrder table)
    # ------------------------------------------------------------------

    async def _load_db_state(self) -> None:
        """Restore pairs from DB on startup."""
        if not self._db_url:
            return
        from src.db.td_orders import load_orders, delete_order
        rows = await load_orders(
            db_url=self._db_url, platform="polymarket",
            strategy_tag=self.strategy_tag,
        )
        now = time.time()
        stale_cutoff = now - 1800
        stale_count = 0
        for row in rows:
            row_ts = row.filled_at or row.placed_at or 0.0
            if row_ts > 0 and row_ts < stale_cutoff:
                stale_count += 1
                self._db_fire(delete_order(db_url=self._db_url, order_id=row.order_id))
                continue

            extra = row.extra or {}
            pair_side = extra.get("pair_side", row.outcome)

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
                pair = self.pairs.get(row.condition_id)
                if not pair:
                    pair = TwoSidedPair(
                        condition_id=row.condition_id,
                        up_token_id="", down_token_id="",
                    )
                    self.pairs[row.condition_id] = pair
                shares = row.shares or row.size_usd / row.price
                if pair_side == "Up" or row.outcome == "Up":
                    pair.up_filled = True
                    pair.up_price = row.price
                    pair.up_shares = shares
                    pair.up_size_usd = row.size_usd
                    pair.up_order_id = row.order_id
                    pair.up_filled_at = row.filled_at or 0.0
                    pair.up_token_id = row.token_id
                else:
                    pair.down_filled = True
                    pair.down_price = row.price
                    pair.down_shares = shares
                    pair.down_size_usd = row.size_usd
                    pair.down_order_id = row.order_id
                    pair.down_filled_at = row.filled_at or 0.0
                    pair.down_token_id = row.token_id
                self._position_order_ids[(row.condition_id, row.outcome)] = row.order_id

        if stale_count:
            logger.info("2sm_stale_orders_cleaned", count=stale_count)

        if self.active_orders or any(p.any_filled for p in self.pairs.values()):
            filled_pairs = sum(1 for p in self.pairs.values() if p.any_filled)
            logger.info(
                "2sm_db_state_loaded",
                orders=len(self.active_orders),
                filled_pairs=filled_pairs,
            )

    def _db_fire(self, coro) -> None:
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
            extra={"pair_side": order.outcome},
        )

    async def _db_mark_filled(self, order_id: str, shares: float, filled_at: float) -> None:
        from src.db.td_orders import mark_filled
        await mark_filled(db_url=self._db_url, order_id=order_id, shares=shares, filled_at=filled_at)

    async def _db_delete_order(self, order_id: str) -> None:
        from src.db.td_orders import delete_order
        await delete_order(db_url=self._db_url, order_id=order_id)

    async def _db_mark_settled(self, order_id: str, pnl: float, settled_at: float) -> None:
        from src.db.td_orders import mark_settled
        await mark_settled(db_url=self._db_url, order_id=order_id, pnl=pnl, settled_at=settled_at)

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

            # Initialize pair if not loaded from DB.
            if cid not in self.pairs:
                self.pairs[cid] = TwoSidedPair(
                    condition_id=cid,
                    up_token_id=clob_ids[0],
                    down_token_id=clob_ids[1],
                )
            else:
                # Fill in token IDs for DB-restored pairs.
                pair = self.pairs[cid]
                if not pair.up_token_id:
                    pair.up_token_id = clob_ids[0]
                if not pair.down_token_id:
                    pair.down_token_id = clob_ids[1]

            try:
                await self.polymarket.subscribe_market(
                    cid, token_map=token_map, send=False,
                )
            except Exception as exc:
                logger.warning("ws_subscribe_failed", condition_id=cid, error=str(exc))

            new_user_markets.append(cid)
            new_count += 1

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

            logger.info(
                "2sm_markets_discovered",
                new=new_count,
                total=len(self.known_markets),
                active_orders=len(self.active_orders),
            )

        await self._prune_expired()

    # ------------------------------------------------------------------
    # Maker loop
    # ------------------------------------------------------------------

    async def maker_loop(self) -> None:
        await asyncio.sleep(3.0)
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

        # Sync book freshness.
        feed_ts = self.polymarket.last_update_ts
        if feed_ts > self._last_book_update:
            self._last_book_update = feed_ts

        # Expire stale pending cancels.
        stale_cancel_ids = [
            oid for oid, order in self._pending_cancels.items()
            if order.cancelled_at > 0 and now - order.cancelled_at > 30.0
        ]
        for oid in stale_cancel_ids:
            self._pending_cancels.pop(oid)
            self._db_fire(self._db_delete_order(oid))

        # Circuit breaker gate.
        if self.guard:
            await self.guard.heartbeat()
            if not await self.guard.is_trading_allowed(last_book_update=self._last_book_update):
                if self.guard.should_cancel_orders and self.active_orders:
                    await self._cancel_all_orders("stale_escalation")
                if self.paper_mode:
                    self._check_fills_paper(now)
                return

        if self.paper_mode:
            self._check_fills_paper(now)

        # Budget check.
        filled_exposure = sum(
            (p.up_size_usd if p.up_filled else 0) + (p.down_size_usd if p.down_filled else 0)
            for p in self.pairs.values()
        )
        pending_exposure = sum(o.size_usd for o in self.active_orders.values())
        budget_left = self.max_total_exposure_usd - filled_exposure - pending_exposure

        # Cancel near-expiry unfilled orders.
        await self._cancel_near_expiry(now)

        # Count active pair cids (markets with at least one active order).
        active_pair_cids = {o.condition_id for o in self.active_orders.values()}
        # Also count pairs with fills (they occupy slots).
        filled_pair_cids = {cid for cid, p in self.pairs.items() if p.any_filled}
        occupied_slots = len(active_pair_cids | filled_pair_cids)

        pending_cancel_cids = {o.condition_id for o in self._pending_cancels.values()}

        place_intents: list[tuple[str, str, str, float]] = []  # (cid, outcome, token_id, price)
        cancel_ids: list[str] = []

        for cid in list(self.known_markets):
            pair = self.pairs.get(cid)
            if not pair:
                continue

            # Skip if pair already has any fill — becomes directional, no new bids.
            if pair.any_filled:
                continue

            # Concurrent pair limit.
            if occupied_slots >= self.max_concurrent_pairs and cid not in active_pair_cids:
                continue

            if cid in pending_cancel_cids:
                continue

            outcomes = self.market_outcomes.get(cid, [])
            if len(outcomes) < 2:
                continue

            # Get best bids on both sides.
            bid_up, _, ask_up, _ = self.polymarket.get_best_levels(cid, outcomes[0])
            bid_down, _, ask_down, _ = self.polymarket.get_best_levels(cid, outcomes[1])

            if bid_up is not None:
                self._last_bids[(cid, outcomes[0])] = bid_up
            if bid_down is not None:
                self._last_bids[(cid, outcomes[1])] = bid_down

            if bid_up is None or bid_down is None:
                continue

            # Core two-sided edge check.
            combined = bid_up + bid_down
            edge = 1.0 - combined

            if edge < self.min_edge:
                # Not enough edge — cancel any existing orders on this market.
                for outcome in outcomes:
                    existing_oid = self._orders_by_cid_outcome.get((cid, outcome))
                    if existing_oid and existing_oid in self.active_orders:
                        cancel_ids.append(existing_oid)
                continue

            # Bid range filter.
            if bid_up > self.max_bid or bid_down > self.max_bid:
                continue
            if bid_up < self.min_bid or bid_down < self.min_bid:
                continue

            # Post-only: don't cross the book.
            if ask_up is not None and bid_up >= ask_up:
                continue
            if ask_down is not None and bid_down >= ask_down:
                continue

            # Place/maintain bids on both sides.
            for i, outcome in enumerate(outcomes):
                bid_price = bid_up if i == 0 else bid_down
                token_id = self.market_tokens.get((cid, outcome))
                if not token_id:
                    continue

                existing_oid = self._orders_by_cid_outcome.get((cid, outcome))
                existing_order = self.active_orders.get(existing_oid) if existing_oid else None

                if existing_order:
                    # Bid-through fill in paper mode.
                    if self.paper_mode and bid_price < existing_order.price:
                        self._process_fill(existing_order, now)
                        self.active_orders.pop(existing_oid, None)
                        if self._orders_by_cid_outcome.get((cid, outcome)) == existing_oid:
                            del self._orders_by_cid_outcome[(cid, outcome)]
                        # DO NOT cancel other side — keep it for two-sided profit.
                    # Otherwise keep existing order (no cancel+replace).
                else:
                    if budget_left >= self.order_size_usd:
                        place_intents.append((cid, outcome, token_id, bid_price))

        # Execute cancels.
        for oid in cancel_ids:
            order = self.active_orders.get(oid)
            if not order:
                continue
            key = (order.condition_id, order.outcome)
            if not self.paper_mode and self.executor:
                order.cancelled_at = now
                self._pending_cancels[oid] = order
                try:
                    await self.executor.cancel_order(oid)
                    self._pending_cancels.pop(oid, None)
                    self.active_orders.pop(oid, None)
                    if self._orders_by_cid_outcome.get(key) == oid:
                        del self._orders_by_cid_outcome[key]
                    self._db_fire(self._db_delete_order(oid))
                except Exception as exc:
                    logger.warning("cancel_failed", order_id=oid[:16], error=str(exc)[:80])
            else:
                self.active_orders.pop(oid, None)
                if self._orders_by_cid_outcome.get(key) == oid:
                    del self._orders_by_cid_outcome[key]
                self._db_fire(self._db_delete_order(oid))

        # Execute placements.
        placed = 0
        pending_cancel_cids = {o.condition_id for o in self._pending_cancels.values()}
        for cid, outcome, token_id, price in place_intents:
            if cid in pending_cancel_cids:
                continue
            # Don't place if already have an order on this side.
            if (cid, outcome) in self._orders_by_cid_outcome:
                continue
            if budget_left < self.order_size_usd:
                continue
            order_id = await self._place_order(cid, outcome, token_id, price, now)
            if order_id:
                placed += 1
                budget_left -= self.order_size_usd

        # Periodic status.
        if now - self._last_status_time >= 30.0:
            self._last_status_time = now
            self._print_status(now)

    # ------------------------------------------------------------------
    # Near-expiry cancel
    # ------------------------------------------------------------------

    async def _cancel_near_expiry(self, now: float) -> None:
        """Cancel unfilled orders on markets approaching resolution."""
        for cid, mkt in list(self.known_markets.items()):
            slug = _first_event_slug(mkt).lower()
            parts = slug.rsplit("-", 1)
            if len(parts) != 2 or not parts[1].isdigit():
                continue
            slot_end = int(parts[1]) + 900  # slot_ts + 15 min
            time_to_close = slot_end - now

            if 0 < time_to_close <= self.cancel_before_close:
                pair = self.pairs.get(cid)
                if not pair:
                    continue
                outcomes = self.market_outcomes.get(cid, [])
                for i, outcome in enumerate(outcomes):
                    is_up = (i == 0)
                    already_filled = (is_up and pair.up_filled) or (not is_up and pair.down_filled)
                    if already_filled:
                        continue
                    oid = self._orders_by_cid_outcome.get((cid, outcome))
                    if oid and oid in self.active_orders:
                        await self._cancel_order(oid)
                        logger.info("near_expiry_cancel", cid=cid[:16], outcome=outcome,
                                    time_to_close=round(time_to_close))

    async def _cancel_order(self, oid: str) -> None:
        """Cancel a single order."""
        order = self.active_orders.get(oid)
        if not order:
            return
        key = (order.condition_id, order.outcome)
        if not self.paper_mode and self.executor:
            order.cancelled_at = time.time()
            self._pending_cancels[oid] = order
            try:
                await self.executor.cancel_order(oid)
                self._pending_cancels.pop(oid, None)
                self.active_orders.pop(oid, None)
                if self._orders_by_cid_outcome.get(key) == oid:
                    del self._orders_by_cid_outcome[key]
                self._db_fire(self._db_delete_order(oid))
            except Exception:
                pass
        else:
            self.active_orders.pop(oid, None)
            if self._orders_by_cid_outcome.get(key) == oid:
                del self._orders_by_cid_outcome[key]
            self._db_fire(self._db_delete_order(oid))

    # ------------------------------------------------------------------
    # Order management
    # ------------------------------------------------------------------

    async def _place_order(
        self, cid: str, outcome: str, token_id: str, price: float, now: float,
    ) -> Optional[str]:
        """Place a GTC maker BUY order at the given price."""
        if not self.manager:
            return None

        temp_id = f"_placing_{cid}_{outcome}_{int(round(price*100))}"
        placeholder = PassiveOrder(
            order_id=temp_id, condition_id=cid, outcome=outcome,
            token_id=token_id, price=price, size_usd=self.order_size_usd,
            placed_at=now,
        )
        self.active_orders[temp_id] = placeholder
        self._orders_by_cid_outcome[(cid, outcome)] = temp_id

        slug = _first_event_slug(self.known_markets.get(cid, {}))
        intent = TradeIntent(
            condition_id=cid, token_id=token_id, outcome=outcome,
            side="BUY", price=price, size_usd=self.order_size_usd,
            reason="2s_maker_passive", title=slug, timestamp=now,
        )
        try:
            pending = await self.manager.place(intent)
        except Exception as exc:
            self.active_orders.pop(temp_id, None)
            if self._orders_by_cid_outcome.get((cid, outcome)) == temp_id:
                del self._orders_by_cid_outcome[(cid, outcome)]
            logger.warning("order_place_exception", error=str(exc)[:80])
            return None

        if not pending.order_id:
            self.active_orders.pop(temp_id, None)
            if self._orders_by_cid_outcome.get((cid, outcome)) == temp_id:
                del self._orders_by_cid_outcome[(cid, outcome)]
            return None

        # Fill listener may have already matched a fill to the placeholder.
        if temp_id not in self.active_orders:
            if self._orders_by_cid_outcome.get((cid, outcome)) == temp_id:
                del self._orders_by_cid_outcome[(cid, outcome)]
            # Save to DB with real ID.
            order = PassiveOrder(
                order_id=pending.order_id, condition_id=cid, outcome=outcome,
                token_id=token_id, price=price, size_usd=self.order_size_usd,
                placed_at=now,
            )
            self._db_fire(self._db_save_order(order))
            pair = self.pairs.get(cid)
            if pair:
                if outcome == self.market_outcomes.get(cid, ["Up", "Down"])[0]:
                    self._db_fire(self._db_mark_filled(pending.order_id, pair.up_shares, now))
                else:
                    self._db_fire(self._db_mark_filled(pending.order_id, pair.down_shares, now))
            self._position_order_ids[(cid, outcome)] = pending.order_id
            logger.info("2sm_fill_during_placement", order_id=pending.order_id[:16])
            return pending.order_id

        # No fill yet — replace placeholder.
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

    PAPER_FILL_TIMEOUT: float = 30.0
    PAPER_FILL_MAX_SPREAD: float = 0.03

    def _check_fills_paper(self, now: float) -> None:
        """Paper mode: simulate maker fills."""
        filled_ids: list[str] = []
        for order_id, order in self.active_orders.items():
            bid, _, ask, _ = self.polymarket.get_best_levels(
                order.condition_id, order.outcome
            )
            # 1. Ask crossed down to our bid.
            if ask is not None and ask <= order.price:
                self._process_fill(order, now)
                filled_ids.append(order_id)
            # 2. Bid dropped below our price — our level consumed.
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
                # DO NOT cancel other side — keep it for two-sided profit.

    def _process_fill(self, order: PassiveOrder, now: float) -> None:
        """Record a fill and update the pair state."""
        shares = order.size_usd / order.price
        pair = self.pairs.get(order.condition_id)
        if not pair:
            return

        outcomes = self.market_outcomes.get(order.condition_id, ["Up", "Down"])
        is_up = (order.outcome == outcomes[0])

        if is_up:
            pair.up_filled = True
            pair.up_price = order.price
            pair.up_shares = shares
            pair.up_size_usd = order.size_usd
            pair.up_order_id = order.order_id
            pair.up_filled_at = now
        else:
            pair.down_filled = True
            pair.down_price = order.price
            pair.down_shares = shares
            pair.down_size_usd = order.size_usd
            pair.down_order_id = order.order_id
            pair.down_filled_at = now

        self._position_order_ids[(order.condition_id, order.outcome)] = order.order_id
        self.total_fills += 1
        self._db_fire(self._db_mark_filled(order.order_id, shares, now))

        if pair.both_filled:
            self.total_both_fills += 1
            logger.info(
                "2sm_pair_complete",
                condition_id=order.condition_id[:16],
                edge=f"{pair.structural_edge:.2%}",
                combined=f"{pair.combined_cost:.3f}",
                paper=self.paper_mode,
            )
        else:
            logger.info(
                "2sm_one_fill",
                condition_id=order.condition_id[:16],
                outcome=order.outcome,
                price=order.price,
                shares=round(shares, 2),
                paper=self.paper_mode,
            )

        # Record via TradeManager.
        if self.manager:
            self.manager._pending.pop(order.order_id, None)
            slug = _first_event_slug(self.known_markets.get(order.condition_id, {}))
            intent = TradeIntent(
                condition_id=order.condition_id, token_id=order.token_id,
                outcome=order.outcome, side="BUY", price=order.price,
                size_usd=order.size_usd, reason="2s_maker_passive",
                title=slug, timestamp=now,
            )
            fill_result = FillResult(filled=True, shares=shares, avg_price=order.price)
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self.manager.record_fill_direct(
                    intent, fill_result,
                    execution_mode="paper_fill" if self.paper_mode else "live_fill",
                    extra_state={
                        "condition_id": order.condition_id,
                        "pair_side": order.outcome,
                        "both_filled": pair.both_filled,
                        "combined_cost": round(pair.combined_cost, 4) if pair.both_filled else 0,
                        "structural_edge": round(pair.structural_edge, 4) if pair.both_filled else 0,
                    },
                ))
            except RuntimeError:
                pass

    async def _fill_listener(self) -> None:
        """Drain fills from WS User channel (live mode)."""
        if not self.user_feed:
            return
        while True:
            try:
                evt: UserTradeEvent = await self.user_feed.fills.get()
            except asyncio.CancelledError:
                break

            matched_order: Optional[PassiveOrder] = None
            matched_id: Optional[str] = None
            source = "active"

            # Priority 1: exact maker_order_id match.
            if evt.maker_order_id:
                if evt.maker_order_id in self.active_orders:
                    matched_id = evt.maker_order_id
                    matched_order = self.active_orders[matched_id]
                elif evt.maker_order_id in self._pending_cancels:
                    matched_id = evt.maker_order_id
                    matched_order = self._pending_cancels[matched_id]
                    source = "pending_cancel"
                    logger.warning(
                        "2sm_late_fill",
                        order_id=matched_id[:16],
                        condition_id=matched_order.condition_id[:16],
                    )

            # Priority 2: placeholder match.
            if not matched_order:
                for oid, order in self.active_orders.items():
                    if oid.startswith("_placing_") and order.condition_id == evt.market:
                        matched_order = order
                        matched_id = oid
                        break

            # Priority 3: placeholder in pending_cancels.
            if not matched_order:
                for oid, order in self._pending_cancels.items():
                    if oid.startswith("_placing_") and order.condition_id == evt.market:
                        matched_order = order
                        matched_id = oid
                        source = "pending_cancel"
                        break

            # Fallback: broad condition_id match.
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
                            break

            if not matched_order or not matched_id:
                logger.warning(
                    "2sm_fill_unmatched",
                    market=evt.market[:16],
                    side=evt.side,
                    price=evt.price,
                )
                continue

            now = time.time()
            self._process_fill(matched_order, now)
            if source == "active":
                self.active_orders.pop(matched_id, None)
            else:
                self._pending_cancels.pop(matched_id, None)
            key = (matched_order.condition_id, matched_order.outcome)
            if self._orders_by_cid_outcome.get(key) == matched_id:
                del self._orders_by_cid_outcome[key]
            # DO NOT cancel other side — two-sided strategy keeps both.

    # ------------------------------------------------------------------
    # Cancel all
    # ------------------------------------------------------------------

    async def _cancel_all_orders(self, reason: str) -> None:
        count = len(self.active_orders)
        for oid in list(self.active_orders):
            order = self.active_orders.pop(oid)
            key = (order.condition_id, order.outcome)
            if self._orders_by_cid_outcome.get(key) == oid:
                del self._orders_by_cid_outcome[key]
            if not self.paper_mode and self.executor:
                order.cancelled_at = time.time()
                self._pending_cancels[oid] = order
                asyncio.create_task(self._async_cancel(oid))
            else:
                self._db_fire(self._db_delete_order(oid))
        logger.warning("2sm_all_orders_cancelled", reason=reason, count=count)

    async def _async_cancel(self, order_id: str) -> None:
        try:
            await self.executor.cancel_order(order_id)
            self._pending_cancels.pop(order_id, None)
            self._db_fire(self._db_delete_order(order_id))
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Settlement
    # ------------------------------------------------------------------

    async def _prune_expired(self) -> None:
        now = time.time()
        to_remove: list[str] = []

        for cid, mkt in self.known_markets.items():
            slug = _first_event_slug(mkt).lower()
            parts = slug.rsplit("-", 1)
            if len(parts) == 2 and parts[1].isdigit():
                slot_end = int(parts[1]) + 900
                if now > slot_end + 300:  # 5 min grace
                    to_remove.append(cid)

        actually_removed: list[str] = []
        for cid in to_remove:
            # Cancel remaining orders.
            oids_to_cancel = [oid for oid, o in self.active_orders.items()
                              if o.condition_id == cid]
            for oid in oids_to_cancel:
                order = self.active_orders.pop(oid)
                self._db_fire(self._db_delete_order(oid))
                if not self.paper_mode and self.executor:
                    try:
                        await self.executor.cancel_order(oid)
                    except Exception:
                        pass
            for outcome in self.market_outcomes.get(cid, []):
                self._orders_by_cid_outcome.pop((cid, outcome), None)

            # Settle pair.
            pair = self.pairs.pop(cid, None)
            if pair and pair.any_filled:
                settled = await self._settle_pair(pair, now)
                if not settled:
                    self.pairs[cid] = pair
                    continue

            actually_removed.append(cid)
            try:
                await self.polymarket.unsubscribe_market(cid)
            except Exception:
                pass
            self.known_markets.pop(cid, None)
            for outcome in self.market_outcomes.pop(cid, []):
                self._last_bids.pop((cid, outcome), None)

        if actually_removed:
            logger.info("2sm_markets_pruned", count=len(actually_removed),
                        remaining=len(self.known_markets))

        # Settle orphans (loaded from DB but market no longer known).
        orphan_cids = [cid for cid in list(self.pairs) if cid not in self.known_markets]
        for cid in orphan_cids:
            pair = self.pairs.pop(cid)
            if pair.any_filled:
                logger.warning("2sm_orphan_settling", condition_id=cid[:16])
                await self._settle_pair(pair, now)

    async def _query_resolution(self, cid: str, outcome: str) -> Optional[bool]:
        """Query resolution for a specific outcome. Returns True/False/None."""
        if not self._http_client:
            return None

        GAMMA_URL = "https://gamma-api.polymarket.com"
        slug = _first_event_slug(self.known_markets.get(cid, {}))

        if slug:
            try:
                resp = await self._http_client.get(
                    f"{GAMMA_URL}/events", params={"slug": slug}, timeout=10.0,
                )
                if resp.status_code == 200:
                    events = resp.json()
                    if events:
                        mkt = events[0].get("markets", [{}])[0]
                        outcome_prices_raw = json.loads(mkt.get("outcomePrices", "[]"))
                        outcomes_raw = json.loads(mkt.get("outcomes", "[]"))
                        for i, o in enumerate(outcomes_raw):
                            if o == outcome and i < len(outcome_prices_raw):
                                price = float(outcome_prices_raw[i])
                                if price >= 0.9:
                                    return True
                                if price <= 0.1:
                                    return False
            except Exception as exc:
                logger.warning("gamma_resolution_failed", cid=cid[:16], error=str(exc)[:60])

        CLOB_URL = settings.POLYMARKET_CLOB_HTTP
        try:
            resp = await self._http_client.get(
                f"{CLOB_URL}/markets/{cid}", timeout=10.0,
            )
            if resp.status_code == 200:
                tokens = resp.json().get("tokens", [])
                for tok in tokens:
                    if tok.get("outcome") == outcome:
                        price = float(tok.get("price", 0.5))
                        if price >= 0.9:
                            return True
                        if price <= 0.1:
                            return False
        except Exception as exc:
            logger.warning("clob_resolution_failed", cid=cid[:16], error=str(exc)[:60])

        return None

    async def _settle_pair(self, pair: TwoSidedPair, now: float) -> bool:
        """Settle a two-sided pair. Returns True if settled, False if deferred."""
        cid = pair.condition_id
        outcomes = self.market_outcomes.get(cid, ["Up", "Down"])

        # Determine resolution: did "Up" win?
        up_outcome = outcomes[0] if outcomes else "Up"
        resolution = await self._query_resolution(cid, up_outcome)

        if resolution is not None:
            up_won = resolution
        else:
            # Fallback: last book state.
            bid_up = self._last_bids.get((cid, outcomes[0]))
            bid_down = self._last_bids.get((cid, outcomes[1])) if len(outcomes) > 1 else None

            if bid_up is not None and bid_up >= 0.9:
                up_won = True
            elif bid_down is not None and bid_down >= 0.9:
                up_won = False
            elif bid_up is not None and bid_up <= 0.1:
                up_won = False
            elif bid_down is not None and bid_down <= 0.1:
                up_won = True
            else:
                # Cannot determine — defer.
                latest_fill = max(
                    pair.up_filled_at if pair.up_filled else 0,
                    pair.down_filled_at if pair.down_filled else 0,
                )
                age_min = (now - latest_fill) / 60 if latest_fill > 0 else 0
                if age_min < 60:
                    logger.warning("2sm_settle_deferred", cid=cid[:16], age_min=round(age_min, 1))
                    return False
                up_won = (bid_up or 0) >= 0.5
                logger.error("2sm_settle_forced", cid=cid[:16], bid_up=bid_up, bid_down=bid_down)

        # Compute PnL.
        total_pnl = 0.0
        if pair.both_filled:
            # Guaranteed profit scenario.
            total_cost = pair.up_size_usd + pair.down_size_usd
            if up_won:
                total_return = pair.up_shares * 1.0
            else:
                total_return = pair.down_shares * 1.0
            total_pnl = total_return - total_cost
            self.total_both_fills += 0  # already counted on fill
        elif pair.up_filled and not pair.down_filled:
            if up_won:
                total_pnl = pair.up_shares * (1.0 - pair.up_price)
            else:
                total_pnl = -pair.up_size_usd
        elif pair.down_filled and not pair.up_filled:
            if not up_won:
                total_pnl = pair.down_shares * (1.0 - pair.down_price)
            else:
                total_pnl = -pair.down_size_usd
        else:
            return True  # Neither filled, nothing to settle.

        won = total_pnl > 0
        if won:
            self.total_wins += 1
        else:
            self.total_losses += 1
        self.realized_pnl += total_pnl

        pair_type = "both" if pair.both_filled else "one_sided"
        logger.info(
            "2sm_settled",
            cid=cid[:16],
            pair_type=pair_type,
            won=won,
            pnl=round(total_pnl, 4),
            total_pnl=round(self.realized_pnl, 4),
            record=f"{self.total_wins}W-{self.total_losses}L",
        )

        # Record settlement for each filled side.
        for i, (filled, price, shares, size_usd, order_id) in enumerate([
            (pair.up_filled, pair.up_price, pair.up_shares, pair.up_size_usd, pair.up_order_id),
            (pair.down_filled, pair.down_price, pair.down_shares, pair.down_size_usd, pair.down_order_id),
        ]):
            if not filled:
                continue
            outcome = outcomes[i] if i < len(outcomes) else ("Up" if i == 0 else "Down")
            side_won = (i == 0 and up_won) or (i == 1 and not up_won)
            side_pnl = shares * (1.0 - price) if side_won else -size_usd

            oid = self._position_order_ids.pop((cid, outcome), order_id)
            if oid:
                self._db_fire(self._db_mark_settled(oid, side_pnl, now))

            if self.manager:
                slug = _first_event_slug(self.known_markets.get(cid, {}))
                token_id = self.market_tokens.get((cid, outcome), "")
                resolution_price = 1.0 if side_won else 0.0
                settle_intent = TradeIntent(
                    condition_id=cid, token_id=token_id,
                    outcome=outcome, side="SELL",
                    price=price, size_usd=size_usd,
                    reason="settlement", title=slug, timestamp=now,
                )
                settle_fill = FillResult(
                    filled=True, shares=shares,
                    avg_price=resolution_price, pnl_delta=side_pnl,
                )
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(self.manager.record_settle_direct(
                        settle_intent, settle_fill,
                        extra_state={
                            "pair_type": pair_type,
                            "combined_cost": round(pair.combined_cost, 4) if pair.both_filled else 0,
                        },
                    ))
                except RuntimeError:
                    pass

        if self.guard:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self.guard.record_result(pnl=total_pnl, won=won))
            except RuntimeError:
                pass

        return True

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def _print_status(self, now: float) -> None:
        exposure = sum(
            (p.up_size_usd if p.up_filled else 0) + (p.down_size_usd if p.down_filled else 0)
            for p in self.pairs.values()
        )
        both_count = sum(1 for p in self.pairs.values() if p.both_filled)
        one_count = sum(1 for p in self.pairs.values() if p.any_filled and not p.both_filled)

        # Price snapshot.
        price_parts: list[str] = []
        for cid in list(self.known_markets)[:4]:
            outcomes = self.market_outcomes.get(cid, [])
            if len(outcomes) >= 2:
                b_up, _, _, _ = self.polymarket.get_best_levels(cid, outcomes[0])
                b_dn, _, _, _ = self.polymarket.get_best_levels(cid, outcomes[1])
                bu = f"{b_up:.2f}" if b_up else "?"
                bd = f"{b_dn:.2f}" if b_dn else "?"
                combined = f"{b_up + b_dn:.2f}" if b_up and b_dn else "?"
                price_parts.append(f"U:{bu}/D:{bd}={combined}")

        print(
            f"[{time.strftime('%H:%M:%S')}] "
            f"cb={'ON' if (self.guard and self.guard.circuit_broken) else 'off'} "
            f"mkts={len(self.known_markets)} "
            f"orders={len(self.active_orders)} "
            f"both={both_count} one={one_count} "
            f"fills={self.total_fills}({self.total_both_fills}x2) "
            f"{self.total_wins}W-{self.total_losses}L "
            f"pnl=${self.realized_pnl:+.2f} "
            f"exp=${exposure:.0f} "
            f"[{' | '.join(price_parts)}]"
        )

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    async def run(self) -> None:
        await self._load_db_state()
        await self.polymarket.connect()

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
                "crypto_2s_maker_started",
                symbols=self.symbols,
                paper=self.paper_mode,
                min_edge=self.min_edge,
                bid_range=f"[{self.min_bid}, {self.max_bid}]",
                order_size=self.order_size_usd,
                max_exposure=self.max_total_exposure_usd,
                max_pairs=self.max_concurrent_pairs,
                cancel_before_close=self.cancel_before_close,
                markets=len(self.known_markets),
                user_ws=self.user_feed is not None,
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
        description="Two-sided passive maker for crypto 15-min markets"
    )
    p.add_argument("--symbols", type=str, default="BTCUSDT,ETHUSDT")
    p.add_argument("--paper", action="store_true", default=True)
    p.add_argument("--live", action="store_true", default=False)
    p.add_argument("--min-edge", type=float, default=0.02,
                    help="Min combined edge: 1.0 - bid_up - bid_down (default: 0.02)")
    p.add_argument("--max-bid", type=float, default=0.60,
                    help="Max bid price per side (default: 0.60)")
    p.add_argument("--min-bid", type=float, default=0.30,
                    help="Min bid price per side (default: 0.30)")
    p.add_argument("--order-size", type=float, default=0.0,
                    help="USD per side per market (0 = derive from wallet)")
    p.add_argument("--max-exposure", type=float, default=0.0,
                    help="Max total USD exposure (0 = derive from wallet)")
    p.add_argument("--max-concurrent", type=int, default=10,
                    help="Max simultaneous two-sided pairs")
    p.add_argument("--cancel-before-close", type=int, default=180,
                    help="Cancel unfilled bids N seconds before resolution")
    p.add_argument("--wallet", type=float, default=0.0,
                    help="Wallet USD (0 = auto-detect)")
    p.add_argument("--discovery-interval", type=float, default=60.0)
    p.add_argument("--maker-interval", type=float, default=0.5)
    p.add_argument("--strategy-tag", type=str, default="crypto_2s_maker")
    p.add_argument("--db-url", type=str, default=settings.DATABASE_URL)
    p.add_argument("--cb-max-losses", type=int, default=5)
    p.add_argument("--cb-max-drawdown", type=float, default=-50.0)
    p.add_argument("--cb-stale-seconds", type=float, default=30.0)
    p.add_argument("--cb-stale-cancel", type=float, default=120.0)
    p.add_argument("--cb-stale-exit", type=float, default=300.0)
    p.add_argument("--cb-daily-limit", type=float, default=-200.0)
    return p


async def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    paper_mode = not args.live
    strategy_tag = args.strategy_tag.strip() or "crypto_2s_maker"

    run_id = f"{strategy_tag}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

    executor: Optional[PolymarketExecutor] = None
    if not paper_mode:
        executor = PolymarketExecutor.from_settings()

    manager = TradeManager(
        executor=executor,
        strategy="CryptoTwoSidedMaker",
        paper=paper_mode,
        db_url=args.db_url,
        event_type=TWO_SIDED_MAKER_EVENT_TYPE,
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
                api_key=api_key, api_secret=api_secret,
                api_passphrase=api_passphrase,
            )

    wallet = args.wallet
    if args.order_size > 0 and args.max_exposure > 0:
        order_size = args.order_size
        max_exposure = args.max_exposure
    else:
        if wallet <= 0 and not paper_mode:
            wallet = await manager.get_wallet_balance()
            if wallet <= 0:
                logger.error("auto_wallet_failed", hint="pass --wallet explicitly")
                return
            logger.info("auto_wallet_detected", wallet_usd=round(wallet, 2))
        order_size = args.order_size if args.order_size > 0 else max(wallet * 0.025, 5.0)
        max_exposure = args.max_exposure if args.max_exposure > 0 else max(wallet * 0.50, 50.0)

    maker = CryptoTwoSidedMaker(
        executor=executor,
        polymarket=polymarket,
        user_feed=user_feed,
        manager=manager,
        symbols=symbols,
        min_edge=args.min_edge,
        max_bid=args.max_bid,
        min_bid=args.min_bid,
        order_size_usd=order_size,
        max_total_exposure_usd=max_exposure,
        max_concurrent_pairs=args.max_concurrent,
        cancel_before_close=args.cancel_before_close,
        paper_mode=paper_mode,
        discovery_interval=args.discovery_interval,
        maker_loop_interval=args.maker_interval,
        strategy_tag=strategy_tag,
        guard=guard,
        db_url=args.db_url,
    )

    wallet_src = "auto" if args.wallet <= 0 else "manual"
    print(f"=== Crypto Two-Sided Maker {'(PAPER)' if paper_mode else '(LIVE)'} ===")
    print(f"  Symbols:       {', '.join(symbols)}")
    print(f"  Wallet:        ${wallet:.0f} ({wallet_src})")
    print(f"  Min edge:      {args.min_edge:.1%}")
    print(f"  Bid range:     [{args.min_bid}, {args.max_bid}]")
    print(f"  Order size:    ${order_size:.2f}/side")
    print(f"  Max exposure:  ${max_exposure:.2f}")
    print(f"  Max pairs:     {args.max_concurrent}")
    print(f"  Cancel before: {args.cancel_before_close}s")
    print(f"  Strategy:      Two-sided passive maker on 15-min crypto markets")
    print()

    await maker.run()


if __name__ == "__main__":
    if uvloop is not None:
        uvloop.install()
    asyncio.run(main())
