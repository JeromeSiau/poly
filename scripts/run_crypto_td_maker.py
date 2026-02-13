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
    ./run run_crypto_td_maker.py --paper          # paper mode (default)
    ./run run_crypto_td_maker.py --live            # real orders
    ./run run_crypto_td_maker.py --target-bid 0.78 # adjust min entry level
"""

from __future__ import annotations

import argparse
import asyncio
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

# Crypto market discovery and parsing utilities.
from src.utils.crypto_markets import CRYPTO_SYMBOL_TO_SLUG, fetch_crypto_markets
from src.utils.parsing import parse_json_list, _to_float, _first_event_slug
from src.execution import TradeManager, TradeIntent, FillResult

logger = structlog.get_logger()

TD_MAKER_EVENT_TYPE = "crypto_td_maker"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class PassiveOrder:
    """A GTC maker bid waiting to be filled."""
    order_id: str
    condition_id: str
    outcome: str  # "Up" or "Down"
    token_id: str
    price: float
    size_usd: float
    placed_at: float


@dataclass(slots=True)
class OpenPosition:
    """A filled passive bid held until market resolution."""
    condition_id: str
    outcome: str
    token_id: str
    entry_price: float
    size_usd: float
    shares: float  # size_usd / entry_price
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
    ) -> None:
        self.executor = executor
        self.polymarket = polymarket
        self.user_feed = user_feed
        self.manager = manager
        self.symbols = symbols
        self.target_bid = target_bid
        self.max_bid = max_bid
        self.order_size_usd = order_size_usd
        self.max_total_exposure_usd = max_total_exposure_usd
        self.paper_mode = paper_mode
        self.discovery_interval = discovery_interval
        self.maker_loop_interval = maker_loop_interval
        self.strategy_tag = strategy_tag

        # State
        self.known_markets: dict[str, dict[str, Any]] = {}  # cid -> raw market
        self.market_tokens: dict[tuple[str, str], str] = {}  # (cid, outcome) -> token_id
        self.market_outcomes: dict[str, list[str]] = {}  # cid -> ["Up", "Down"]
        self.active_orders: dict[str, PassiveOrder] = {}  # order_id -> order
        self._orders_by_cid_outcome: dict[tuple[str, str], str] = {}
        self.positions: dict[str, OpenPosition] = {}  # cid -> position
        self._http_client: Optional[httpx.AsyncClient] = None
        self._cycle_count: int = 0
        self._last_status_time: float = 0.0
        # Last known bid per (cid, outcome) — fallback for settlement when book empties.
        self._last_bids: dict[tuple[str, str], float] = {}

        # Stats
        self.total_fills: int = 0
        self.total_wins: int = 0
        self.total_losses: int = 0
        self.realized_pnl: float = 0.0

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

            # Subscribe to orderbook WS.
            try:
                await self.polymarket.subscribe_market(cid, token_map=token_map)
            except Exception as exc:
                logger.warning("ws_subscribe_failed", condition_id=cid, error=str(exc))

            if self.user_feed and self.user_feed.is_connected:
                try:
                    await self.user_feed.subscribe_markets([cid])
                except Exception:
                    pass

            new_count += 1

        if new_count:
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

        # Check for paper fills every tick.
        if self.paper_mode:
            self._check_fills_paper(now)

        # Check exposure budget.
        current_exposure = sum(p.size_usd for p in self.positions.values())
        pending_exposure = sum(o.size_usd for o in self.active_orders.values())
        budget_left = self.max_total_exposure_usd - current_exposure - pending_exposure

        cancel_ids: list[str] = []
        place_intents: list[tuple[str, str, str, float]] = []  # (cid, outcome, token_id, bid_price)

        for cid in list(self.known_markets):
            # Skip if already have a position on this market.
            if cid in self.positions:
                continue

            outcomes = self.market_outcomes.get(cid, [])
            if len(outcomes) < 2:
                continue

            for outcome in outcomes:
                token_id = self.market_tokens.get((cid, outcome))
                if not token_id:
                    continue

                bid, bid_sz, ask, ask_sz = self.polymarket.get_best_levels(cid, outcome)
                if bid is not None:
                    self._last_bids[(cid, outcome)] = bid
                existing_key = (cid, outcome)
                existing_oid = self._orders_by_cid_outcome.get(existing_key)
                existing_order = self.active_orders.get(existing_oid) if existing_oid else None

                # Is the bid in our target range?
                bid_in_range = (
                    bid is not None
                    and self.target_bid <= bid <= self.max_bid
                )

                if existing_order:
                    if not bid_in_range:
                        # Price left our range → cancel.
                        cancel_ids.append(existing_order.order_id)
                    elif existing_order.price != bid:
                        # Bid moved within range → cancel and re-place at new bid.
                        # Paper fill check: if bid dropped past our price, it's a fill.
                        if self.paper_mode and bid < existing_order.price:
                            self._process_fill(existing_order, now)
                            oid = existing_order.order_id
                            self.active_orders.pop(oid, None)
                            if self._orders_by_cid_outcome.get(existing_key) == oid:
                                del self._orders_by_cid_outcome[existing_key]
                            self._cancel_other_side(cid, outcome)
                        else:
                            cancel_ids.append(existing_order.order_id)
                            if budget_left >= self.order_size_usd:
                                place_intents.append((cid, outcome, token_id, bid))
                    # else: order still at correct price, do nothing.
                elif bid_in_range and budget_left >= self.order_size_usd:
                    # No existing order, price is in range → place new bid.
                    place_intents.append((cid, outcome, token_id, bid))

        # Execute cancels.
        for oid in cancel_ids:
            order = self.active_orders.pop(oid, None)
            if order:
                key = (order.condition_id, order.outcome)
                if self._orders_by_cid_outcome.get(key) == oid:
                    del self._orders_by_cid_outcome[key]
                if not self.paper_mode and self.executor:
                    try:
                        await self.executor.cancel_order(oid)
                    except Exception:
                        pass

        # Execute placements.
        placed = 0
        for cid, outcome, token_id, bid_price in place_intents:
            # Re-check we don't already have position (could have filled above).
            if cid in self.positions:
                continue
            if (cid, outcome) in self._orders_by_cid_outcome:
                continue

            order_id = await self._place_order(cid, outcome, token_id, bid_price, now)
            if order_id:
                placed += 1
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
                f"mkts={len(self.known_markets)} "
                f"orders={len(self.active_orders)} "
                f"pos={len(self.positions)} "
                f"fills={self.total_fills} "
                f"{self.total_wins}W-{self.total_losses}L "
                f"pnl=${self.realized_pnl:+.2f} "
                f"exp=${exposure:.0f} "
                f"prices=[{' '.join(price_parts)}]"
            )

    # ------------------------------------------------------------------
    # Order management
    # ------------------------------------------------------------------

    async def _place_order(
        self, cid: str, outcome: str, token_id: str, price: float, now: float
    ) -> Optional[str]:
        """Place a GTC buy at the given price via TradeManager."""
        if not self.manager:
            return None
        slug = _first_event_slug(self.known_markets.get(cid, {}))
        intent = TradeIntent(
            condition_id=cid, token_id=token_id, outcome=outcome,
            side="BUY", price=price, size_usd=self.order_size_usd,
            reason="td_maker_passive", title=slug, timestamp=now,
        )
        pending = await self.manager.place(intent)
        if not pending.order_id:
            return None
        order = PassiveOrder(
            order_id=pending.order_id, condition_id=cid, outcome=outcome,
            token_id=token_id, price=price, size_usd=self.order_size_usd,
            placed_at=now,
        )
        self.active_orders[pending.order_id] = order
        self._orders_by_cid_outcome[(cid, outcome)] = pending.order_id
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

        1. Ask crossed: ask <= our bid price (original).
        2. Bid-through: bid dropped below our price — our level was consumed.
        3. Time-at-bid: order at best bid with tight spread for PAPER_FILL_TIMEOUT
           seconds — simulates queue priority on an active book.
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
                self._cancel_other_side(order.condition_id, order.outcome)

    def _process_fill(self, order: PassiveOrder, now: float) -> None:
        """Record a filled order as an open position."""
        shares = order.size_usd / order.price
        pos = OpenPosition(
            condition_id=order.condition_id,
            outcome=order.outcome,
            token_id=order.token_id,
            entry_price=order.price,
            size_usd=order.size_usd,
            shares=shares,
            filled_at=now,
        )
        self.positions[order.condition_id] = pos
        self.total_fills += 1

        logger.info(
            "td_order_filled",
            condition_id=order.condition_id[:16],
            outcome=order.outcome,
            price=order.price,
            shares=round(shares, 2),
            total_fills=self.total_fills,
            paper=self.paper_mode,
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
                shares=shares,
                avg_price=order.price,
            )
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self.manager.record_fill_direct(
                    intent, fill_result,
                    execution_mode="paper_fill" if self.paper_mode else "live_fill",
                    extra_state={"condition_id": order.condition_id},
                ))
            except RuntimeError:
                pass

    def _cancel_other_side(self, cid: str, filled_outcome: str) -> None:
        """Cancel the unfilled side's order after one side fills."""
        outcomes = self.market_outcomes.get(cid, [])
        for outcome in outcomes:
            if outcome == filled_outcome:
                continue
            key = (cid, outcome)
            oid = self._orders_by_cid_outcome.pop(key, None)
            if oid and oid in self.active_orders:
                self.active_orders.pop(oid)
                if not self.paper_mode and self.executor:
                    asyncio.create_task(self._async_cancel(oid))
                logger.debug("td_other_side_cancelled", outcome=outcome)

    async def _async_cancel(self, order_id: str) -> None:
        try:
            await self.executor.cancel_order(order_id)
        except Exception:
            pass

    async def _fill_listener(self) -> None:
        """Drain fills from WS User channel (live mode real-time detection)."""
        if not self.user_feed:
            return
        while True:
            try:
                evt: UserTradeEvent = await self.user_feed.fills.get()
            except asyncio.CancelledError:
                break

            matched_order: Optional[PassiveOrder] = None
            matched_id: Optional[str] = None
            for oid, order in self.active_orders.items():
                if order.token_id == evt.asset_id and order.condition_id == evt.market:
                    matched_order = order
                    matched_id = oid
                    break

            if not matched_order or not matched_id:
                continue

            now = time.time()
            self._process_fill(matched_order, now)
            del self.active_orders[matched_id]
            key = (matched_order.condition_id, matched_order.outcome)
            if self._orders_by_cid_outcome.get(key) == matched_id:
                del self._orders_by_cid_outcome[key]
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

        for cid in to_remove:
            # Cancel any remaining orders.
            for outcome in self.market_outcomes.get(cid, []):
                key = (cid, outcome)
                oid = self._orders_by_cid_outcome.pop(key, None)
                if oid:
                    order = self.active_orders.pop(oid, None)
                    if order and not self.paper_mode and self.executor:
                        try:
                            await self.executor.cancel_order(oid)
                        except Exception:
                            pass

            # Settle position if held.
            pos = self.positions.pop(cid, None)
            if pos:
                self._settle_position(pos, now)

            # Unsubscribe.
            try:
                await self.polymarket.unsubscribe_market(cid)
            except Exception:
                pass

            self.known_markets.pop(cid, None)
            for outcome in self.market_outcomes.pop(cid, []):
                self._last_bids.pop((cid, outcome), None)

        if to_remove:
            logger.info("td_markets_pruned", count=len(to_remove), remaining=len(self.known_markets))

    def _settle_position(self, pos: OpenPosition, now: float) -> None:
        """Determine win/loss from last book state and record PnL."""
        bid, _, _, _ = self.polymarket.get_best_levels(pos.condition_id, pos.outcome)
        if bid is None:
            bid = self._last_bids.get((pos.condition_id, pos.outcome))
        if bid is not None and bid >= 0.5:
            won = True
            pnl = pos.shares * (1.0 - pos.entry_price)
        else:
            won = False
            pnl = -pos.size_usd

        if won:
            self.total_wins += 1
        else:
            self.total_losses += 1
        self.realized_pnl += pnl

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

        # Manager handles DB persistence + Telegram notification
        if self.manager:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self.manager.settle(
                    pos.condition_id, pos.outcome, 1.0 if won else 0.0, won,
                ))
            except RuntimeError:
                pass

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    async def run(self) -> None:
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
                "crypto_td_maker_started",
                symbols=self.symbols,
                paper=self.paper_mode,
                target_bid=self.target_bid,
                max_bid=self.max_bid,
                order_size=self.order_size_usd,
                max_exposure=self.max_total_exposure_usd,
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
        description="Passive time-decay maker for crypto 15-min markets"
    )
    p.add_argument("--symbols", type=str, default="BTCUSDT,ETHUSDT")
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
    p.add_argument("--order-size", type=float, default=10.0, help="USD per order")
    p.add_argument("--max-exposure", type=float, default=200.0, help="Max total USD exposure")
    p.add_argument("--discovery-interval", type=float, default=60.0)
    p.add_argument("--maker-interval", type=float, default=0.5, help="Maker loop tick interval")
    p.add_argument("--strategy-tag", type=str, default="crypto_td_maker")
    p.add_argument("--db-url", type=str, default=settings.DATABASE_URL)
    return p


async def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
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
        notify_fills=True,
        notify_closes=True,
    )

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

    maker = CryptoTDMaker(
        executor=executor,
        polymarket=polymarket,
        user_feed=user_feed,
        manager=manager,
        symbols=symbols,
        target_bid=args.target_bid,
        max_bid=args.max_bid,
        order_size_usd=args.order_size,
        max_total_exposure_usd=args.max_exposure,
        paper_mode=paper_mode,
        discovery_interval=args.discovery_interval,
        maker_loop_interval=args.maker_interval,
        strategy_tag=strategy_tag,
    )

    print(f"=== Crypto TD Maker {'(PAPER)' if paper_mode else '(LIVE)'} ===")
    print(f"  Symbols:     {', '.join(symbols)}")
    print(f"  Bid range:   [{args.target_bid}, {args.max_bid}]")
    print(f"  Order size:  ${args.order_size}")
    print(f"  Max exposure: ${args.max_exposure}")
    print(f"  Strategy:    Watch books, bid at current bid when in range")
    print(f"               Hold filled positions to market resolution")
    print()

    await maker.run()


if __name__ == "__main__":
    if uvloop is not None:
        uvloop.install()
    asyncio.run(main())
