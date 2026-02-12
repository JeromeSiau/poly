#!/usr/bin/env python3
"""WebSocket-driven crypto maker for Polymarket 15-min binary markets.

Architecture:
    asyncio.gather(
        binance_feed.listen(),         # BG: real-time BTC/ETH prices
        maker.market_discovery_loop(), # BG: discover new 15-min slots
        maker.maker_loop(),            # Main: read WS state, manage GTC orders
    )

Places GTC limit orders at the bid on both sides (Up + Down) of each
15-minute crypto market.  When both legs fill, cost = bid_up + bid_down ≈ 0.99,
payout = $1.00, profit ≈ $0.01 per complete set.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import httpx
import structlog

try:
    import uvloop
except ImportError:
    uvloop = None

_project_root = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root))

from config.settings import settings
from src.arb.polymarket_executor import PolymarketExecutor
from src.arb.two_sided_inventory import (
    FillResult,
    MarketSnapshot,
    OutcomeQuote,
    TradeIntent,
    TwoSidedInventoryEngine,
)
from src.execution import TradeManager, TradeIntent as ExecTradeIntent
from src.feeds.binance import BinanceFeed
from src.feeds.polymarket import PolymarketFeed, PolymarketUserFeed, UserTradeEvent

# Reuse crypto market discovery from the two-sided runner.
from scripts.run_two_sided_inventory import (
    CRYPTO_SYMBOL_TO_SLUG,
    fetch_crypto_markets,
    parse_json_list,
    _to_float,
    _first_event_slug,
)

logger = structlog.get_logger()

CRYPTO_MAKER_EVENT_TYPE = "crypto_maker"

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ActiveOrder:
    """A GTC order placed on the CLOB (or simulated in paper mode)."""
    order_id: str
    intent: TradeIntent
    placed_at: float
    condition_id: str
    outcome: str
    token_id: str


# ---------------------------------------------------------------------------
# CryptoMaker
# ---------------------------------------------------------------------------

class CryptoMaker:
    """WebSocket-driven market maker for crypto 15-min binary markets."""

    def __init__(
        self,
        *,
        engine: TwoSidedInventoryEngine,
        executor: Optional[PolymarketExecutor],
        binance: BinanceFeed,
        polymarket: PolymarketFeed,
        user_feed: Optional[PolymarketUserFeed],
        manager: Optional[TradeManager] = None,
        symbols: list[str],
        paper_mode: bool = True,
        maker_loop_interval: float = 0.5,
        discovery_interval: float = 60.0,
        fill_check_interval: float = 3.0,
        max_order_usd: float = 10.0,
        min_order_usd: float = 1.0,
        max_outcome_inv_usd: float = 25.0,
        min_pair_profit: float = 0.01,
        strategy_tag: str = "crypto_maker",
    ) -> None:
        self.engine = engine
        self.executor = executor
        self.binance = binance
        self.polymarket = polymarket
        self.user_feed = user_feed
        self.manager = manager
        self.symbols = symbols
        self.paper_mode = paper_mode
        self.maker_loop_interval = maker_loop_interval
        self.discovery_interval = discovery_interval
        self.fill_check_interval = fill_check_interval
        self.max_order_usd = max_order_usd
        self.min_order_usd = min_order_usd
        self.max_outcome_inv_usd = max_outcome_inv_usd
        self.min_pair_profit = min_pair_profit
        self.strategy_tag = strategy_tag

        # State
        self.active_orders: dict[str, ActiveOrder] = {}
        self._orders_by_key: dict[tuple[str, str, str], str] = {}  # (cid,outcome,side)->order_id
        self.known_markets: dict[str, dict[str, Any]] = {}   # condition_id -> raw market
        self.market_symbol: dict[str, str] = {}               # condition_id -> "BTCUSDT"
        self.market_outcomes: dict[str, list[str]] = {}       # condition_id -> ["Up","Down"]
        self.market_tokens: dict[tuple[str, str], str] = {}   # (cid, outcome) -> token_id
        self._last_fill_check: float = 0.0
        self._http_client: Optional[httpx.AsyncClient] = None
        self._paper_order_counter: int = 0
        self._cycle_count: int = 0

    # ------------------------------------------------------------------
    # Market discovery
    # ------------------------------------------------------------------

    async def market_discovery_loop(self) -> None:
        """Periodically discover new 15-min crypto markets and subscribe."""
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

            # Determine which CEX symbol this market tracks.
            slug = _first_event_slug(mkt).lower()
            for sym, prefix in CRYPTO_SYMBOL_TO_SLUG.items():
                if slug.startswith(prefix):
                    self.market_symbol[cid] = sym
                    break

            # Subscribe to RTDS orderbook for this market's tokens.
            try:
                await self.polymarket.subscribe_market(cid, token_map=token_map)
            except Exception as exc:
                logger.warning("ws_subscribe_failed", condition_id=cid, error=str(exc))

            # Subscribe User channel for real-time fill notifications.
            if self.user_feed and self.user_feed.is_connected:
                try:
                    await self.user_feed.subscribe_markets([cid])
                except Exception as exc:
                    logger.warning("user_ws_subscribe_failed", condition_id=cid, error=str(exc))

            new_count += 1

        if new_count:
            logger.info(
                "markets_discovered",
                new=new_count,
                total=len(self.known_markets),
                symbols=self.symbols,
            )

        # Prune expired markets (slot ended > 5 min ago).
        now = time.time()
        to_remove = []
        for cid, mkt in self.known_markets.items():
            slug = _first_event_slug(mkt).lower()
            # Extract slot timestamp from slug: "btc-updown-15m-1707576600"
            parts = slug.rsplit("-", 1)
            if len(parts) == 2 and parts[1].isdigit():
                slot_end = int(parts[1]) + 900  # slot_ts + 15 min
                if now > slot_end + 300:  # 5 min grace
                    to_remove.append(cid)

        settled_count = 0
        for cid in to_remove:
            # Cancel any active orders on this market.
            for oid, order in list(self.active_orders.items()):
                if order.condition_id == cid:
                    if not self.paper_mode and self.executor:
                        try:
                            await self.executor.cancel_order(oid)
                        except Exception:
                            pass
                    key = (order.condition_id, order.outcome, order.intent.side)
                    if self._orders_by_key.get(key) == oid:
                        del self._orders_by_key[key]
                    del self.active_orders[oid]

            # Settle open positions before removing market data.
            settled_count += await self._settle_market(cid, now)

            try:
                await self.polymarket.unsubscribe_market(cid)
            except Exception:
                pass
            if self.user_feed and self.user_feed.is_connected:
                try:
                    await self.user_feed.unsubscribe_markets([cid])
                except Exception:
                    pass
            self.known_markets.pop(cid, None)
            self.market_symbol.pop(cid, None)
            self.market_outcomes.pop(cid, None)

        if to_remove:
            logger.info(
                "markets_pruned",
                count=len(to_remove),
                settled=settled_count,
                remaining=len(self.known_markets),
            )

    # ------------------------------------------------------------------
    # Settlement
    # ------------------------------------------------------------------

    async def _settle_market(self, condition_id: str, now: float) -> int:
        """Settle all open positions for an expired market.

        Strategy:
        1. Pair merge first — if both outcomes held, merge at $1.00 total.
        2. Settle remaining — winner at $1.00, loser at $0.00 based on
           last known book (the winning outcome's bid ≈ 1.0 after resolution).

        Returns number of positions settled.
        """
        outcomes = self.market_outcomes.get(condition_id, [])
        if len(outcomes) < 2:
            return 0

        inv = self.engine.get_open_inventory()
        by_outcome = inv.get(condition_id)
        if not by_outcome:
            return 0

        mkt = self.known_markets.get(condition_id, {})
        slug = _first_event_slug(mkt)
        settled = 0

        # --- Phase 1: pair merge (both sides held → profit = $1 - cost) ---
        states = {o: by_outcome.get(o) for o in outcomes if by_outcome.get(o) and by_outcome[o].shares > 0}
        if len(states) == 2:
            out_a, out_b = outcomes[0], outcomes[1]
            merge_shares = min(states[out_a].shares, states[out_b].shares)
            if merge_shares > 0:
                for outcome in [out_a, out_b]:
                    fill = self.engine.settle_position(
                        condition_id=condition_id,
                        outcome=outcome,
                        settlement_price=0.50,
                        timestamp=now,
                    )
                    if fill.shares > 0:
                        settled += 1
                        await self._persist_settlement(
                            condition_id, outcome, slug, fill, 0.50, now,
                        )
                pair_cost = states[out_a].avg_price + states[out_b].avg_price
                logger.info(
                    "maker_pair_settled",
                    condition_id=condition_id[:16],
                    shares=merge_shares,
                    cost=round(pair_cost, 4),
                    profit=round(1.0 - pair_cost, 4),
                )

        # --- Phase 2: settle any remaining single-side positions ---
        inv = self.engine.get_open_inventory()
        by_outcome = inv.get(condition_id, {})
        for outcome in outcomes:
            state = by_outcome.get(outcome)
            if not state or state.shares <= 0:
                continue

            # Determine settlement from last book: bid near 1.0 = winner.
            bid, _, _, _ = self.polymarket.get_best_levels(condition_id, outcome)
            if bid is not None and bid >= 0.5:
                settlement_price = 1.0
            else:
                settlement_price = 0.0

            fill = self.engine.settle_position(
                condition_id=condition_id,
                outcome=outcome,
                settlement_price=settlement_price,
                timestamp=now,
            )
            if fill.shares > 0:
                settled += 1
                await self._persist_settlement(
                    condition_id, outcome, slug, fill, settlement_price, now,
                )
                logger.info(
                    "maker_position_settled",
                    condition_id=condition_id[:16],
                    outcome=outcome,
                    settlement=settlement_price,
                    shares=fill.shares,
                    pnl=round(fill.realized_pnl_delta, 4),
                )

        return settled

    async def _persist_settlement(
        self,
        condition_id: str,
        outcome: str,
        slug: str,
        fill: FillResult,
        settlement_price: float,
        now: float,
    ) -> None:
        """Persist a settlement via TradeManager."""
        if self.manager is None:
            return
        won = settlement_price >= 0.5
        try:
            await self.manager.settle(
                condition_id=condition_id,
                outcome=outcome,
                settlement_price=settlement_price,
                won=won,
            )
        except Exception as exc:
            logger.warning("persist_settlement_failed", error=str(exc))

    # ------------------------------------------------------------------
    # Maker loop
    # ------------------------------------------------------------------

    async def maker_loop(self) -> None:
        """Event-driven loop: wake on book update or timeout."""
        # Wait for initial discovery.
        await asyncio.sleep(2.0)

        while True:
            # Wake on WS book update OR max interval — whichever comes first.
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

        # Fill check: paper mode runs every tick (free, no I/O);
        # live mode throttled to fill_check_interval (HTTP polling).
        if self.paper_mode or now - self._last_fill_check >= self.fill_check_interval:
            await self._check_fills(now)
            self._last_fill_check = now

        # Single inventory snapshot for the entire tick.
        inv = self.engine.get_open_inventory()

        # Phase 1: collect cancels, paper fills, and new intents (no I/O yet).
        cancel_ids: list[str] = []
        paper_fill_ids: list[str] = []
        place_intents: list[TradeIntent] = []

        for cid, mkt in list(self.known_markets.items()):
            symbol = self.market_symbol.get(cid)
            if not symbol:
                continue

            outcomes = self.market_outcomes.get(cid)
            if not outcomes or len(outcomes) < 2:
                continue

            # Read both outcomes' books up front for pair cost check.
            book: dict[str, tuple] = {}  # outcome -> (bid, bid_sz, ask, ask_sz)
            for outcome in outcomes:
                levels = self.polymarket.get_best_levels(cid, outcome)
                if levels[0] is not None:
                    book[outcome] = levels

            # GUARD 1: Require BOTH books present.  Without both sides we
            # cannot verify pair profitability and risk naked directional
            # exposure.  Cancel any existing orders and skip.
            if len(book) < 2:
                for outcome in outcomes:
                    existing = self._find_order(cid, outcome, "BUY")
                    if existing is not None:
                        cancel_ids.append(existing.order_id)
                continue

            # GUARD 2: Pair cost filter — bid_up + bid_down must be
            # < $1.00 - min_pair_profit.
            pair_cost = sum(v[0] for v in book.values())
            fee_cost = pair_cost * self.engine.fee_pct
            if pair_cost + fee_cost >= 1.0 - self.min_pair_profit:
                for outcome in outcomes:
                    existing = self._find_order(cid, outcome, "BUY")
                    if existing is not None:
                        cancel_ids.append(existing.order_id)
                continue

            # Compute per-outcome inventory for pair imbalance check.
            cid_inv = inv.get(cid, {})
            inv_shares: dict[str, float] = {}
            for outcome in outcomes:
                out_state = cid_inv.get(outcome)
                inv_shares[outcome] = out_state.shares if out_state else 0.0

            for outcome in outcomes:
                if outcome not in book:
                    continue
                bid, bid_sz, ask, ask_sz = book[outcome]

                token_id = self.market_tokens.get((cid, outcome))
                if not token_id:
                    continue

                fair = (bid + ask) / 2 if ask is not None else bid + 0.005

                # O(1) lookup via index.
                existing = self._find_order(cid, outcome, "BUY")

                if existing is not None:
                    if existing.intent.price != bid:
                        # Paper mode: bid dropped past our price → someone sold
                        # through our level → treat as fill, not cancel.
                        if self.paper_mode and bid < existing.intent.price:
                            paper_fill_ids.append(existing.order_id)
                        else:
                            cancel_ids.append(existing.order_id)
                        existing = None

                if existing is None:
                    edge = fair - bid - (bid * self.engine.fee_pct)
                    if edge < self.engine.min_edge_pct:
                        continue

                    # GUARD 3: Pair imbalance — don't buy more of one side
                    # if we already hold much more than the other side.
                    # Max imbalance = one order worth of shares.
                    other = [o for o in outcomes if o != outcome][0]
                    my_shares = inv_shares.get(outcome, 0.0)
                    other_shares = inv_shares.get(other, 0.0)
                    max_imbalance = self.max_order_usd / bid if bid > 0 else 0
                    if my_shares - other_shares > max_imbalance:
                        continue

                    # Size: respect inventory limits (using cached inv).
                    current_inv = 0.0
                    out_state = cid_inv.get(outcome)
                    if out_state:
                        current_inv = out_state.shares * out_state.avg_price
                    remaining = self.max_outcome_inv_usd - current_inv
                    if remaining <= self.min_order_usd:
                        continue

                    size_usd = min(self.max_order_usd, remaining)
                    if bid_sz is not None:
                        size_usd = min(size_usd, bid_sz * bid)
                    if size_usd < self.min_order_usd:
                        continue

                    place_intents.append(TradeIntent(
                        condition_id=cid,
                        title=str(mkt.get("question", "")),
                        outcome=outcome,
                        token_id=token_id,
                        side="BUY",
                        price=bid,
                        size_usd=size_usd,
                        edge_pct=edge,
                        reason="maker_ws_bid",
                        timestamp=now,
                    ))

        # Phase 2a: process paper fills (bid dropped through our level).
        for oid in paper_fill_ids:
            order = self.active_orders.pop(oid, None)
            if order is None:
                continue
            key = (order.condition_id, order.outcome, order.intent.side)
            if self._orders_by_key.get(key) == oid:
                del self._orders_by_key[key]
            fill = self.engine.apply_fill(order.intent)
            logger.info(
                "maker_order_filled",
                order_id=oid,
                condition_id=order.condition_id,
                outcome=order.outcome,
                side=order.intent.side,
                price=order.intent.price,
                age_seconds=round(now - order.placed_at, 1),
                paper=True,
                trigger="bid_drop",
            )

        # Phase 2b: execute cancels in parallel.
        if cancel_ids:
            await asyncio.gather(
                *(self._cancel_order(oid) for oid in cancel_ids),
                return_exceptions=True,
            )

        # Phase 3: execute placements in parallel.
        orders_placed = 0
        if place_intents:
            results = await asyncio.gather(
                *(self._place_order(intent) for intent in place_intents),
                return_exceptions=True,
            )
            orders_placed = sum(1 for r in results if r and not isinstance(r, BaseException))

        # Periodic status log.
        if self._cycle_count % 20 == 0:
            n_positions = sum(len(by_out) for by_out in inv.values())
            print(
                f"[{time.strftime('%H:%M:%S')}] markets={len(self.known_markets)} "
                f"active_orders={len(self.active_orders)} positions={n_positions} "
                f"placed={orders_placed} cancelled={len(cancel_ids)} "
                f"fills={len(paper_fill_ids)} "
                f"realized_pnl=${self.engine.get_realized_pnl():,.4f}"
            )

    # ------------------------------------------------------------------
    # Order management
    # ------------------------------------------------------------------

    def _find_order(self, condition_id: str, outcome: str, side: str) -> Optional[ActiveOrder]:
        order_id = self._orders_by_key.get((condition_id, outcome, side))
        return self.active_orders.get(order_id) if order_id else None

    async def _place_order(self, intent: TradeIntent) -> Optional[str]:
        """Place a GTC order via TradeManager. Returns order_id or None."""
        if not self.manager:
            return None

        exec_intent = ExecTradeIntent(
            condition_id=intent.condition_id,
            token_id=intent.token_id,
            outcome=intent.outcome,
            side=intent.side,
            price=intent.price,
            size_usd=intent.size_usd,
            reason=intent.reason,
            title=intent.title,
            edge_pct=intent.edge_pct,
            timestamp=intent.timestamp,
        )
        pending = await self.manager.place(exec_intent)
        if not pending.order_id:
            return None
        order_id = pending.order_id

        active = ActiveOrder(
            order_id=order_id,
            intent=intent,
            placed_at=intent.timestamp,
            condition_id=intent.condition_id,
            outcome=intent.outcome,
            token_id=intent.token_id,
        )
        self.active_orders[order_id] = active
        self._orders_by_key[(intent.condition_id, intent.outcome, intent.side)] = order_id
        logger.info(
            "maker_order_placed",
            order_id=order_id,
            condition_id=intent.condition_id,
            outcome=intent.outcome,
            side=intent.side,
            price=intent.price,
            size=intent.size_usd,
            paper=self.paper_mode,
        )
        return order_id

    async def _cancel_order(self, order_id: str) -> None:
        """Cancel an active order."""
        order = self.active_orders.pop(order_id, None)
        if order is None:
            return

        key = (order.condition_id, order.outcome, order.intent.side)
        if self._orders_by_key.get(key) == order_id:
            del self._orders_by_key[key]

        if not self.paper_mode and self.executor:
            try:
                await self.executor.cancel_order(order_id)
            except Exception as exc:
                logger.warning("cancel_failed", order_id=order_id, error=str(exc))

        logger.debug(
            "maker_order_cancelled",
            order_id=order_id,
            condition_id=order.condition_id,
            outcome=order.outcome,
        )

    async def _check_fills(self, now: float) -> None:
        """Check pending orders for fills."""
        if self.paper_mode:
            await self._check_fills_paper(now)
        else:
            await self._check_fills_live(now)

    async def _check_fills_paper(self, now: float) -> None:
        """Simulate fills: order fills when WS ask crosses our bid price."""
        filled_ids: list[str] = []
        for order_id, order in self.active_orders.items():
            bid, bid_sz, ask, ask_sz = self.polymarket.get_best_levels(
                order.condition_id, order.outcome
            )
            filled = False
            if order.intent.side == "BUY" and ask is not None:
                filled = ask <= order.intent.price
            elif order.intent.side == "SELL" and bid is not None:
                filled = bid >= order.intent.price

            if filled:
                fill = self.engine.apply_fill(order.intent)
                filled_ids.append(order_id)
                logger.info(
                    "maker_order_filled",
                    order_id=order_id,
                    condition_id=order.condition_id,
                    outcome=order.outcome,
                    side=order.intent.side,
                    price=order.intent.price,
                    age_seconds=round(now - order.placed_at, 1),
                    paper=True,
                )

        for oid in filled_ids:
            order = self.active_orders.pop(oid, None)
            if order is not None:
                key = (order.condition_id, order.outcome, order.intent.side)
                if self._orders_by_key.get(key) == oid:
                    del self._orders_by_key[key]

    async def _check_fills_live(self, now: float) -> None:
        """Check live order status via HTTP API (parallel queries)."""
        if not self.executor:
            return

        markets_to_check = list({o.condition_id for o in self.active_orders.values()})
        if not markets_to_check:
            return

        # Parallel HTTP queries for all markets at once.
        results = await asyncio.gather(
            *(self.executor.get_open_orders(m) for m in markets_to_check),
            return_exceptions=True,
        )

        live_order_ids: set[str] = set()
        for i, result in enumerate(results):
            if isinstance(result, BaseException):
                logger.warning("get_orders_failed", market=markets_to_check[i], error=str(result))
                continue
            for o in result:
                oid = o.get("id") or o.get("orderID") or ""
                if oid:
                    live_order_ids.add(oid)

        # Any active order not in live_order_ids is filled or cancelled.
        for order_id, order in list(self.active_orders.items()):
            if order_id not in live_order_ids:
                fill = self.engine.apply_fill(order.intent)
                logger.info(
                    "maker_order_filled",
                    order_id=order_id,
                    condition_id=order.condition_id,
                    outcome=order.outcome,
                    side=order.intent.side,
                    price=order.intent.price,
                    age_seconds=round(now - order.placed_at, 1),
                    paper=False,
                )
                key = (order.condition_id, order.outcome, order.intent.side)
                if self._orders_by_key.get(key) == order_id:
                    del self._orders_by_key[key]
                del self.active_orders[order_id]

    async def _fill_listener(self) -> None:
        """Drain fills from WS User channel in real time (replaces HTTP polling)."""
        if not self.user_feed:
            return
        while True:
            try:
                evt: UserTradeEvent = await self.user_feed.fills.get()
            except asyncio.CancelledError:
                break

            # Find matching active order by checking maker_orders or taker_order.
            matched_order: Optional[ActiveOrder] = None
            matched_id: Optional[str] = None
            for oid, order in self.active_orders.items():
                if order.token_id == evt.asset_id and order.condition_id == evt.market:
                    matched_order = order
                    matched_id = oid
                    break

            if not matched_order or not matched_id:
                continue

            fill = self.engine.apply_fill(matched_order.intent)
            logger.info(
                "maker_ws_fill_detected",
                order_id=matched_id,
                condition_id=matched_order.condition_id,
                outcome=matched_order.outcome,
                side=matched_order.intent.side,
                price=matched_order.intent.price,
                status=evt.status,
            )

            key = (matched_order.condition_id, matched_order.outcome, matched_order.intent.side)
            if self._orders_by_key.get(key) == matched_id:
                del self._orders_by_key[key]
            del self.active_orders[matched_id]

    def _build_snapshot(self, condition_id: str) -> Optional[MarketSnapshot]:
        """Build a MarketSnapshot from WS book state for a condition."""
        outcomes = self.market_outcomes.get(condition_id, [])
        if len(outcomes) < 2:
            return None

        outcome_quotes: dict[str, OutcomeQuote] = {}
        for outcome in outcomes:
            token_id = self.market_tokens.get((condition_id, outcome), "")
            bid, bid_sz, ask, ask_sz = self.polymarket.get_best_levels(condition_id, outcome)
            outcome_quotes[outcome] = OutcomeQuote(
                outcome=outcome,
                token_id=token_id,
                bid=bid,
                ask=ask,
                bid_size=bid_sz,
                ask_size=ask_sz,
            )

        mkt = self.known_markets.get(condition_id, {})
        return MarketSnapshot(
            condition_id=condition_id,
            title=str(mkt.get("question", "")),
            slug=_first_event_slug(mkt),
            outcome_order=outcomes,
            outcomes=outcome_quotes,
            timestamp=time.time(),
            liquidity=_to_float(mkt.get("liquidityNum")),
            volume_24h=_to_float(mkt.get("volume24hr")),
        )

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Connect feeds and run all loops concurrently."""
        await self.binance.connect()
        await self.polymarket.connect()

        # Connect User WS channel for real-time fill detection (live mode).
        if self.user_feed:
            try:
                await self.user_feed.connect()
                # When we have real-time fills, relax HTTP polling to a safety-net (30s).
                self.fill_check_interval = max(self.fill_check_interval, 30.0)
                logger.info("user_ws_connected")
            except Exception as exc:
                logger.warning("user_ws_connect_failed", error=str(exc))
                self.user_feed = None

        timeout = httpx.Timeout(20.0, connect=10.0)
        limits = httpx.Limits(max_connections=20, max_keepalive_connections=10)
        async with httpx.AsyncClient(timeout=timeout, limits=limits, http2=True) as client:
            self._http_client = client
            # Initial discovery before starting loops.
            await self._discover_markets()

            logger.info(
                "crypto_maker_started",
                symbols=self.symbols,
                paper=self.paper_mode,
                markets=len(self.known_markets),
                maker_interval=self.maker_loop_interval,
                min_pair_profit=self.min_pair_profit,
                user_ws=self.user_feed is not None,
            )

            tasks: list[Any] = [
                self.binance.listen(),
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
                await self.binance.disconnect()
                if self.manager:
                    await self.manager.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="WebSocket-driven crypto maker")
    p.add_argument("--symbols", type=str, default="BTCUSDT,ETHUSDT")
    p.add_argument("--paper", action="store_true", default=True, help="Paper trading (no real orders)")
    p.add_argument("--live", action="store_true", default=False, help="Live trading with real GTC orders")
    p.add_argument("--maker-interval", type=float, default=0.5, help="Maker loop interval in seconds")
    p.add_argument("--discovery-interval", type=float, default=60.0, help="Market discovery interval")
    p.add_argument("--fill-check-interval", type=float, default=3.0, help="Fill check polling interval")
    p.add_argument("--min-edge", type=float, default=0.001, help="Minimum edge to place order")
    p.add_argument("--min-pair-profit", type=float, default=0.01, help="Min profit per $1 pair (bid_up+bid_down < 1-this)")
    p.add_argument("--min-order", type=float, default=1.0, help="Minimum order USD")
    p.add_argument("--max-order", type=float, default=10.0, help="Maximum order USD")
    p.add_argument("--max-outcome-inv", type=float, default=25.0, help="Max inventory per outcome USD")
    p.add_argument("--max-market-net", type=float, default=12.0, help="Max net exposure per market USD")
    p.add_argument("--max-hold-seconds", type=float, default=900.0, help="Max hold before stale exit")
    p.add_argument("--strategy-tag", type=str, default="crypto_maker")
    p.add_argument("--db-url", type=str, default=settings.DATABASE_URL)
    return p


async def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    paper_mode = not args.live
    strategy_tag = args.strategy_tag.strip() or "crypto_maker"
    run_id = f"{strategy_tag}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

    engine = TwoSidedInventoryEngine(
        min_edge_pct=args.min_edge,
        exit_edge_pct=args.min_edge / 2,
        min_order_usd=args.min_order,
        max_order_usd=args.max_order,
        max_outcome_inventory_usd=args.max_outcome_inv,
        max_market_net_usd=args.max_market_net,
        max_hold_seconds=args.max_hold_seconds,
        fee_bps=settings.POLYMARKET_FEE_BPS,
        maker_mode=True,
    )

    executor: Optional[PolymarketExecutor] = None
    if not paper_mode:
        executor = PolymarketExecutor(
            host=settings.POLYMARKET_CLOB_HTTP,
            chain_id=settings.POLYMARKET_CHAIN_ID,
            private_key=settings.POLYMARKET_PRIVATE_KEY,
            funder=settings.POLYMARKET_WALLET_ADDRESS,
            api_key=settings.POLYMARKET_API_KEY or None,
            api_secret=settings.POLYMARKET_API_SECRET or None,
            api_passphrase=settings.POLYMARKET_API_PASSPHRASE or None,
        )

    manager = TradeManager(
        executor=executor,
        strategy="CryptoMaker",
        paper=paper_mode,
        db_url=args.db_url,
        event_type=CRYPTO_MAKER_EVENT_TYPE,
        run_id=run_id,
        notify_bids=True,
        notify_fills=True,
        notify_closes=True,
    )

    binance = BinanceFeed(symbols=symbols)
    polymarket = PolymarketFeed()

    # User WS channel for real-time fill detection (live mode only, requires API creds).
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

    maker = CryptoMaker(
        engine=engine,
        executor=executor,
        binance=binance,
        polymarket=polymarket,
        user_feed=user_feed,
        manager=manager,
        symbols=symbols,
        paper_mode=paper_mode,
        maker_loop_interval=args.maker_interval,
        discovery_interval=args.discovery_interval,
        fill_check_interval=args.fill_check_interval,
        max_order_usd=args.max_order,
        min_order_usd=args.min_order,
        max_outcome_inv_usd=args.max_outcome_inv,
        min_pair_profit=args.min_pair_profit,
        strategy_tag=strategy_tag,
    )

    await maker.run()


if __name__ == "__main__":
    if uvloop is not None:
        uvloop.install()
    asyncio.run(main())
