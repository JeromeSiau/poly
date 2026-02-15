"""Last-Penny Sniper Engine.

Buys quasi-certain outcomes (ask >= 0.95) on Polymarket and holds to resolution.
Inspired by Sharky6999 (99.3% win rate, $597K profit).

Architecture:
    MarketScanner finds targets via REST polling (Gamma API + CLOB book)
    SniperEngine subscribes WebSocket for live orderbook on targets
    On ask at price >= threshold → taker BUY via PolymarketExecutor
    Hold to resolution (auto-redeemed by separate run_auto_redeem.py)
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any, Optional

import httpx
import structlog

from src.execution import TradeManager, TradeIntent, FillResult
from src.feeds.polymarket import PolymarketFeed, PolymarketUserFeed
from src.feeds.polymarket_scanner import MarketScanner, SniperTarget
from src.risk.guard import RiskGuard
from src.utils.parsing import parse_json_list

logger = structlog.get_logger()

GAMMA_EVENTS_API = "https://gamma-api.polymarket.com/events"


@dataclass(slots=True)
class SniperPosition:
    """An open sniper position awaiting resolution."""

    condition_id: str
    token_id: str
    outcome: str
    entry_price: float
    shares: float
    size_usd: float
    fee_cost: float
    entry_ts: float
    question: str
    category: str
    slug: str


class SniperEngine:
    """Core sniper loop: scan → subscribe → snipe → hold → settle."""

    def __init__(
        self,
        *,
        polymarket: PolymarketFeed,
        user_feed: Optional[PolymarketUserFeed] = None,
        manager: TradeManager,
        guard: RiskGuard,
        scanner: MarketScanner,
        capital: float = 500.0,
        risk_pct: float = 0.01,
        max_per_market_pct: float = 0.05,
        scan_interval: float = 15.0,
        snipe_loop_interval: float = 1.0,
        settle_check_interval: float = 60.0,
        paper: bool = True,
    ) -> None:
        self.polymarket = polymarket
        self.user_feed = user_feed
        self.manager = manager
        self.guard = guard
        self.scanner = scanner

        self.capital = capital
        self.risk_pct = risk_pct
        self.max_per_market_pct = max_per_market_pct
        self.scan_interval = scan_interval
        self.snipe_loop_interval = snipe_loop_interval
        self.settle_check_interval = settle_check_interval
        self.paper = paper

        # State
        self._positions: dict[str, SniperPosition] = {}
        self._subscribed_markets: set[str] = set()
        self._total_exposure: float = 0.0
        self._session_pnl: float = 0.0
        self._total_wins: int = 0
        self._total_losses: int = 0

    @property
    def positions(self) -> dict[str, SniperPosition]:
        return dict(self._positions)

    @property
    def total_exposure(self) -> float:
        return self._total_exposure

    async def run(self) -> None:
        """Main entry: connect feeds, run scan + snipe + settle loops."""
        logger.info(
            "sniper_starting",
            paper=self.paper,
            min_price=self.scanner.min_price,
            capital=self.capital,
            risk_pct=self.risk_pct,
        )

        await self.polymarket.connect()
        if self.user_feed:
            try:
                await self.user_feed.connect()
                logger.info("sniper_user_ws_connected")
            except Exception as exc:
                logger.warning("sniper_user_ws_failed", error=str(exc))
                self.user_feed = None

        timeout = httpx.Timeout(20.0, connect=10.0)
        limits = httpx.Limits(max_connections=30, max_keepalive_connections=15)
        async with httpx.AsyncClient(
            timeout=timeout, limits=limits,
            headers={"User-Agent": "Mozilla/5.0"},
            follow_redirects=True,
        ) as client:
            self._http_client = client

            # Reload open positions from previous runs
            await self._reload_open_positions(client)

            tasks = [
                self._scan_loop(client),
                self._snipe_loop(),
                self._settle_loop(client),
            ]
            if self.user_feed:
                tasks.append(self._fill_listener())

            try:
                await asyncio.gather(*tasks)
            finally:
                self._http_client = None

    # ------------------------------------------------------------------
    # Reload open positions from DB (survive restarts)
    # ------------------------------------------------------------------

    async def _reload_open_positions(self, client: httpx.AsyncClient) -> None:
        """Reload open sniper positions from the trades API.

        On restart, in-memory positions are lost. This queries the DB
        for is_open=true sniper trades and reconstructs SniperPosition
        objects so the settle loop can close them.

        Retries a few times since the trades API may not be ready yet
        at startup (403/connection errors).
        """
        data = None
        # Use a separate client without proxy for localhost calls —
        # the process may inherit HTTP_PROXY that blocks localhost.
        async with httpx.AsyncClient(trust_env=False, timeout=10.0) as local:
            for attempt in range(5):
                try:
                    resp = await local.get(
                        "http://localhost:8788/trades",
                        params={
                            "event_type": "last_penny_sniper",
                            "is_open": "true",
                            "hours": 168,  # 7 days
                            "limit": 2000,
                        },
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        break
                    logger.warning(
                        "reload_positions_api_error",
                        status=resp.status_code,
                        attempt=attempt + 1,
                        body=resp.text[:200],
                    )
                except Exception as exc:
                    logger.warning(
                        "reload_positions_failed",
                        error=str(exc),
                        attempt=attempt + 1,
                    )
                await asyncio.sleep(3.0)

        if data is None:
            logger.error("reload_positions_gave_up", attempts=5)
            return

        trades = data.get("trades", [])
        loaded = 0
        for t in trades:
            cid = t.get("match_id", "")
            if not cid or cid in self._positions:
                continue

            # game_state fields (from extra_state if available)
            gs = t.get("game_state") or {}
            token_id = gs.get("token_id", "")
            slug = gs.get("slug", "")
            category = gs.get("category", "other")
            fee_cost = gs.get("fee_cost", 0.0)
            shares = gs.get("shares", 0.0)
            size_usd = t.get("size", 0.0)

            # Fallback: if no token_id in game_state, look up via CLOB
            if not token_id:
                token_id = await self._lookup_token_id(client, cid, t.get("outcome", ""))
                if not token_id:
                    continue

            entry_price = t.get("entry_price", 0.0)
            if not shares and entry_price > 0:
                shares = size_usd / entry_price

            self._positions[cid] = SniperPosition(
                condition_id=cid,
                token_id=token_id,
                outcome=t.get("outcome", ""),
                entry_price=entry_price,
                shares=shares,
                size_usd=size_usd,
                fee_cost=fee_cost,
                entry_ts=time.time(),
                question=t.get("title", ""),
                category=category,
                slug=slug,
            )
            self._total_exposure += size_usd
            self.scanner.mark_traded(cid)
            loaded += 1

        if loaded:
            logger.info("sniper_positions_reloaded", count=loaded, exposure=round(self._total_exposure, 2))

    async def _lookup_token_id(
        self, client: httpx.AsyncClient, condition_id: str, outcome: str,
    ) -> str:
        """Look up token_id from CLOB API for legacy trades without extra_state."""
        try:
            resp = await client.get(
                f"https://clob.polymarket.com/markets/{condition_id}",
                timeout=10.0,
            )
            if resp.status_code != 200:
                return ""
            data = resp.json()
            for tok in data.get("tokens", []):
                if tok.get("outcome") == outcome:
                    return tok.get("token_id", "")
        except Exception:
            pass
        return ""

    # ------------------------------------------------------------------
    # Scan loop: discover markets via REST
    # ------------------------------------------------------------------

    async def _scan_loop(self, client: httpx.AsyncClient) -> None:
        """Periodically scan for new sniper targets."""
        while True:
            try:
                targets = await self.scanner.scan(client)

                # Subscribe WebSocket for targets + watchlist (early entry)
                new_count = 0
                for t in list(targets) + self.scanner.watchlist:
                    if t.condition_id not in self._subscribed_markets:
                        token_map = {t.outcome: t.token_id}
                        await self.polymarket.subscribe_market(
                            t.condition_id, token_map, send=False
                        )
                        self._subscribed_markets.add(t.condition_id)
                        new_count += 1

                if new_count:
                    await self.polymarket.flush_subscriptions()
                    logger.info("sniper_subscribed_new", count=new_count)

                logger.info(
                    "sniper_scan_summary",
                    targets=len(targets),
                    watching=len(self.scanner.watchlist),
                    subscribed=len(self._subscribed_markets),
                    positions=len(self._positions),
                    exposure=round(self._total_exposure, 2),
                    pnl=round(self._session_pnl, 2),
                    record=f"{self._total_wins}W-{self._total_losses}L",
                )

            except Exception as exc:
                logger.error("sniper_scan_error", error=str(exc))

            await asyncio.sleep(self.scan_interval)

    # ------------------------------------------------------------------
    # Snipe loop: monitor orderbooks, execute buys
    # ------------------------------------------------------------------

    async def _snipe_loop(self) -> None:
        """Event-driven loop: check orderbooks on every WS update."""
        while True:
            try:
                await asyncio.wait_for(
                    self.polymarket.book_updated.wait(),
                    timeout=self.snipe_loop_interval,
                )
            except asyncio.TimeoutError:
                pass
            self.polymarket.book_updated.clear()

            await self._snipe_tick()

    async def _snipe_tick(self) -> None:
        """Check all targets for snipeable asks."""
        # Skip staleness check before any WS subscriptions exist —
        # last_update_ts is 0.0 until the first message arrives.
        if not self._subscribed_markets:
            return

        if not await self.guard.is_trading_allowed(
            last_book_update=self.polymarket.last_update_ts
        ):
            return

        # Check all watched markets (targets + watchlist) — WS may show
        # a watchlist market crossing min_price between REST scans.
        all_watched = list(self.scanner.targets) + self.scanner.watchlist
        for target in all_watched:
            if target.condition_id in self._positions:
                continue

            # Get live orderbook from WebSocket
            bid, bid_sz, ask, ask_sz = self.polymarket.get_best_levels(
                target.condition_id, target.outcome
            )

            if ask is None or ask < self.scanner.min_price:
                continue
            if ask_sz is None or ask_sz <= 0:
                continue

            # Sizing: constant risk per trade
            available_capital = self.capital - self._total_exposure
            if available_capital < 5.0:
                return

            max_risk = available_capital * self.risk_pct
            max_market = available_capital * self.max_per_market_pct

            loss_per_share = ask + target.fee_pct
            if loss_per_share <= 0:
                continue

            max_shares = max_risk / loss_per_share
            order_usd = min(
                max_shares * ask,
                max_market,
                ask_sz * ask,
            )

            if order_usd < 1.0:
                continue

            shares = order_usd / ask

            intent = TradeIntent(
                condition_id=target.condition_id,
                token_id=target.token_id,
                outcome=target.outcome,
                side="BUY",
                price=ask,
                size_usd=order_usd,
                reason="sniper_entry",
                title=target.question[:80],
                edge_pct=(1.0 - ask) * 100,
            )

            try:
                pending = await self.manager.place(intent)
                if not pending.order_id:
                    logger.warning("sniper_order_rejected", condition_id=target.condition_id[:16])
                    continue

                logger.info(
                    "sniper_order_placed",
                    condition_id=target.condition_id[:16],
                    outcome=target.outcome,
                    price=ask,
                    size_usd=round(order_usd, 2),
                    shares=round(shares, 1),
                    category=target.category,
                    question=target.question[:60],
                )

                fee_cost = shares * target.fee_pct
                self._positions[target.condition_id] = SniperPosition(
                    condition_id=target.condition_id,
                    token_id=target.token_id,
                    outcome=target.outcome,
                    entry_price=ask,
                    shares=shares,
                    size_usd=order_usd,
                    fee_cost=fee_cost,
                    entry_ts=time.time(),
                    question=target.question,
                    category=target.category,
                    slug=target.slug,
                )
                self._total_exposure += order_usd
                self.scanner.mark_traded(target.condition_id)

                # Taker = immediate fill
                fill = FillResult(filled=True, shares=shares, avg_price=ask)
                extra = {
                    "token_id": target.token_id,
                    "slug": target.slug,
                    "category": target.category,
                    "fee_cost": round(fee_cost, 6),
                }
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(self.manager.record_fill_direct(
                        intent, fill,
                        execution_mode="paper" if self.paper else "live",
                        extra_state=extra,
                    ))
                except RuntimeError:
                    pass

            except Exception as exc:
                logger.error(
                    "sniper_order_error",
                    condition_id=target.condition_id[:16],
                    error=str(exc),
                )

    # ------------------------------------------------------------------
    # Settlement loop: check resolved markets
    # ------------------------------------------------------------------

    async def _settle_loop(self, client: httpx.AsyncClient) -> None:
        """Periodically check if any held positions have resolved."""
        while True:
            await asyncio.sleep(self.settle_check_interval)

            settled_cids: list[str] = []
            for cid, pos in list(self._positions.items()):
                resolved = await self._check_resolution(client, pos)
                if resolved is None:
                    continue

                won = resolved
                pnl = (pos.shares * (1.0 - pos.entry_price) - pos.fee_cost
                       if won else -(pos.size_usd + pos.fee_cost))

                if won:
                    self._total_wins += 1
                else:
                    self._total_losses += 1
                self._session_pnl += pnl
                self._total_exposure -= pos.size_usd

                await self.guard.record_result(pnl=pnl, won=won)

                logger.info(
                    "sniper_settled",
                    condition_id=cid[:16],
                    outcome=pos.outcome,
                    entry_price=pos.entry_price,
                    won=won,
                    pnl=round(pnl, 4),
                    total_pnl=round(self._session_pnl, 2),
                    record=f"{self._total_wins}W-{self._total_losses}L",
                )

                # Record settlement via TradeManager
                settle_intent = TradeIntent(
                    condition_id=cid,
                    token_id=pos.token_id,
                    outcome=pos.outcome,
                    side="SELL",
                    price=pos.entry_price,
                    size_usd=pos.size_usd,
                    reason="settlement",
                    title=pos.question[:80],
                )
                settle_fill = FillResult(
                    filled=True,
                    shares=pos.shares,
                    avg_price=1.0 if won else 0.0,
                    pnl_delta=pnl,
                )
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(self.manager.record_settle_direct(
                        settle_intent, settle_fill,
                    ))
                except RuntimeError:
                    pass

                settled_cids.append(cid)

            for cid in settled_cids:
                self._positions.pop(cid, None)
                try:
                    await self.polymarket.unsubscribe_market(cid)
                except Exception:
                    pass

            if settled_cids:
                logger.info(
                    "sniper_settle_batch",
                    settled=len(settled_cids),
                    remaining=len(self._positions),
                )

    async def _check_resolution(
        self, client: httpx.AsyncClient, pos: SniperPosition,
    ) -> Optional[bool]:
        """Query Gamma API for market resolution. Returns True/False/None."""
        if not pos.slug:
            return await self._check_resolution_clob(client, pos)

        try:
            resp = await client.get(
                GAMMA_EVENTS_API,
                params={"slug": pos.slug},
                timeout=10.0,
            )
            if resp.status_code != 200:
                return None
            events = resp.json()
            if not events:
                return None

            for mkt in events[0].get("markets", []):
                if str(mkt.get("conditionId", "")) != pos.condition_id:
                    continue
                if not mkt.get("closed"):
                    continue
                outcome_prices = parse_json_list(mkt.get("outcomePrices", []))
                outcomes = parse_json_list(mkt.get("outcomes", []))
                for i, outcome in enumerate(outcomes):
                    if str(outcome) == pos.outcome and i < len(outcome_prices):
                        price = float(outcome_prices[i])
                        if price >= 0.9:
                            return True
                        if price <= 0.1:
                            return False
        except Exception:
            pass

        return None

    async def _check_resolution_clob(
        self, client: httpx.AsyncClient, pos: SniperPosition,
    ) -> Optional[bool]:
        """Fallback: check CLOB API for resolution."""
        try:
            resp = await client.get(
                f"https://clob.polymarket.com/markets/{pos.condition_id}",
                timeout=10.0,
            )
            if resp.status_code != 200:
                return None
            data = resp.json()
            if not data.get("closed"):
                return None
            tokens = data.get("tokens", [])
            for tok in tokens:
                if tok.get("outcome") == pos.outcome:
                    price = float(tok.get("price", 0.5))
                    if price >= 0.9:
                        return True
                    if price <= 0.1:
                        return False
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    # Fill listener (live mode)
    # ------------------------------------------------------------------

    async def _fill_listener(self) -> None:
        """Drain fills from WS User channel (live mode)."""
        if not self.user_feed:
            return
        while True:
            try:
                evt = await self.user_feed.fills.get()
                logger.info(
                    "sniper_fill_event",
                    order_id=evt.order_id[:16],
                    market=evt.market[:16],
                    price=evt.price,
                    size=evt.size,
                    status=evt.status,
                )
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("sniper_fill_listener_error", error=str(exc))
                await asyncio.sleep(1)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        """Return current sniper stats for dashboard / logging."""
        return {
            "positions": len(self._positions),
            "total_exposure": round(self._total_exposure, 2),
            "available_capital": round(self.capital - self._total_exposure, 2),
            "session_pnl": round(self._session_pnl, 2),
            "wins": self._total_wins,
            "losses": self._total_losses,
            "subscribed_markets": len(self._subscribed_markets),
        }
