#!/usr/bin/env python3
"""Passive time-decay maker for Kalshi crypto bracket markets.

STRATEGY (validated by historical calibration on 777K markets):
    On Kalshi hourly crypto bracket markets (KXBTCD/KXETHD), the
    favourite side (YES priced 75-85c) wins ~93-96% of the time vs
    the 75-85% implied — a +10-14% edge BEFORE fees.

    After taker fees (~2.5% at 80c) or maker fees (~1.25%), the edge
    remains massive: +11% taker, +13% maker.

    For each new hourly event:
      1. Fetch all strikes, find the one with YES ~= TARGET_BID (e.g. 75c).
      2. Place a GTC limit BUY YES at TARGET_BID on that strike.
      3. Hold to resolution. Win ~93% → net +24% per fill.
      4. If not filled → no cost.

REQUIRES:
    pip install kalshi-python
    Kalshi API key (RSA key pair) configured in .env:
      KALSHI_API_KEY_ID=...
      KALSHI_PRIVATE_KEY_PATH=...  (path to PEM file)

USAGE:
    ./run run_kalshi_td_maker.py --paper          # paper mode (default)
    ./run run_kalshi_td_maker.py --live            # real orders
    ./run run_kalshi_td_maker.py --target-bid 78   # 78 cents
"""

from __future__ import annotations

import argparse
import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import structlog

from src.utils.logging import configure_logging

configure_logging()

from config.settings import settings
from src.execution import TradeManager, TradeIntent
from src.feeds.kalshi_executor import (
    KalshiExecutor,
    KALSHI_API_BASE,
    KALSHI_DEMO_API_BASE,
)

logger = structlog.get_logger()

# Crypto event ticker prefixes
KALSHI_CRYPTO_PREFIXES = ["KXBTCD", "KXETHD"]

KALSHI_TD_MAKER_EVENT_TYPE = "kalshi_td_maker"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class KalshiStrike:
    """A single strike within a crypto bracket event."""
    ticker: str
    event_ticker: str
    title: str
    subtitle: str  # e.g. "$98,000 or above"
    yes_bid: int  # cents
    yes_ask: int  # cents
    no_bid: int
    no_ask: int
    last_price: int  # cents
    volume: int
    open_interest: int
    close_time: str
    status: str


@dataclass(slots=True)
class OpenPosition:
    """A filled passive bid held until settlement."""
    ticker: str
    event_ticker: str
    entry_price_cents: int
    count: int
    filled_at: float


# ---------------------------------------------------------------------------
# KalshiTDMaker
# ---------------------------------------------------------------------------

class KalshiTDMaker:
    """Passive time-decay maker for Kalshi crypto bracket markets.

    Scans hourly crypto events, finds the strike with YES price closest
    to target_bid, places a GTC limit buy, and holds to settlement.

    Uses KalshiExecutor for API calls and TradeManager for persistence
    and Telegram notifications.
    """

    def __init__(
        self,
        *,
        executor: KalshiExecutor,
        manager: TradeManager,
        target_bid_cents: int = 75,
        min_bid_cents: int = 65,
        max_bid_cents: int = 85,
        order_size_contracts: int = 10,
        max_total_exposure_usd: float = 200.0,
        paper_mode: bool = True,
        scan_interval: float = 120.0,
        symbols: list[str] | None = None,
    ) -> None:
        self.executor = executor
        self.manager = manager
        self.target_bid_cents = target_bid_cents
        self.min_bid_cents = min_bid_cents
        self.max_bid_cents = max_bid_cents
        self.order_size_contracts = order_size_contracts
        self.max_total_exposure_usd = max_total_exposure_usd
        self.paper_mode = paper_mode
        self.scan_interval = scan_interval
        self.symbols = symbols or ["KXBTCD", "KXETHD"]

        # State
        self.known_events: set[str] = set()
        self.positions: dict[str, OpenPosition] = {}  # ticker -> position

        # Stats (mirror from manager for status display)
        self.total_fills: int = 0

    # ------------------------------------------------------------------
    # Market scanning
    # ------------------------------------------------------------------

    async def scan_loop(self) -> None:
        """Periodically scan for new crypto events and place bids."""
        while True:
            try:
                await self._scan_and_bid()
            except Exception as exc:
                logger.error("kalshi_scan_error", error=str(exc))
            await asyncio.sleep(self.scan_interval)

    async def _scan_and_bid(self) -> None:
        """Find new hourly crypto events and place passive bids on best strikes."""
        for prefix in self.symbols:
            try:
                data = await self.executor.api_get("/markets", params={
                    "event_ticker": prefix,
                    "status": "open",
                    "limit": 200,
                })
                if not data:
                    continue

                markets = data.get("markets", [])
                if not markets:
                    continue

                # Group by event_ticker
                events: dict[str, list[dict]] = {}
                for mkt in markets:
                    et = mkt.get("event_ticker", "")
                    if et:
                        events.setdefault(et, []).append(mkt)

                new_events = 0
                for event_ticker, strikes in events.items():
                    if event_ticker in self.known_events:
                        continue
                    self.known_events.add(event_ticker)
                    new_events += 1

                    # Find the best strike: YES price closest to target_bid
                    best = self._find_best_strike(strikes)
                    if best:
                        await self._place_bid(best)

                if new_events:
                    logger.info(
                        "kalshi_events_scanned",
                        prefix=prefix,
                        new_events=new_events,
                        pending=len(self.manager.get_pending_orders()),
                        positions=len(self.positions),
                    )

            except Exception as exc:
                logger.warning("kalshi_prefix_scan_error", prefix=prefix, error=str(exc))

    def _find_best_strike(self, strikes: list[dict]) -> Optional[dict]:
        """Find the strike with YES ask closest to target_bid within range."""
        best: Optional[dict] = None
        best_dist = 999

        for mkt in strikes:
            yes_ask = mkt.get("yes_ask", 0) or 0
            if yes_ask < self.min_bid_cents or yes_ask > self.max_bid_cents:
                continue

            dist = abs(yes_ask - self.target_bid_cents)
            if dist < best_dist:
                best_dist = dist
                best = mkt

        return best

    async def _place_bid(self, strike: dict) -> None:
        """Place a GTC limit buy YES on the selected strike via TradeManager."""
        ticker = strike.get("ticker", "")
        event_ticker = strike.get("event_ticker", "")

        if ticker in self.positions:
            return

        # Check exposure limit
        current_exposure = sum(
            p.count * p.entry_price_cents / 100.0 for p in self.positions.values()
        )
        if current_exposure >= self.max_total_exposure_usd:
            return

        yes_ask = strike.get("yes_ask", 0) or 0
        price_01 = self.target_bid_cents / 100.0
        size_usd = self.order_size_contracts * price_01

        intent = TradeIntent(
            condition_id=event_ticker,
            token_id=ticker,
            outcome="yes",
            side="BUY",
            price=price_01,
            size_usd=size_usd,
            reason="kalshi_td_maker",
            title=f"{event_ticker} YES @{self.target_bid_cents}c x{self.order_size_contracts}",
            edge_pct=0.0,
        )

        pending = await self.manager.place(intent)

        if not pending.order_id:
            return

        # In paper mode, simulate immediate fill if ask <= target
        if self.paper_mode:
            if yes_ask <= self.target_bid_cents and yes_ask > 0:
                pos = OpenPosition(
                    ticker=ticker,
                    event_ticker=event_ticker,
                    entry_price_cents=yes_ask,
                    count=self.order_size_contracts,
                    filled_at=time.time(),
                )
                self.positions[ticker] = pos
                self.total_fills += 1
                logger.info(
                    "kalshi_paper_fill",
                    ticker=ticker,
                    price_cents=yes_ask,
                    count=self.order_size_contracts,
                )
                return

        logger.info(
            "kalshi_bid_placed",
            ticker=ticker,
            price_cents=self.target_bid_cents,
            count=self.order_size_contracts,
            paper=self.paper_mode,
        )

    # ------------------------------------------------------------------
    # Fill & settlement checking
    # ------------------------------------------------------------------

    async def fill_check_loop(self) -> None:
        """Periodically check for fills and settlements."""
        await asyncio.sleep(5.0)
        while True:
            try:
                if not self.paper_mode:
                    await self._check_fills_live()
                await self._check_settlements()
            except Exception as exc:
                logger.error("kalshi_fill_check_error", error=str(exc))
            await asyncio.sleep(30.0)

    async def _check_fills_live(self) -> None:
        """Check if any pending bids have been filled via API."""
        pending = self.manager.get_pending_orders()
        if not pending:
            return

        # Build a reverse map: ticker -> order_id for matching fills
        ticker_to_oid: dict[str, str] = {}
        for oid, p in pending.items():
            ticker_to_oid[p.intent.token_id] = oid

        try:
            data = await self.executor.api_get("/portfolio/fills", params={"limit": 100})
            if not data:
                return
            fills = data.get("fills", [])
        except Exception:
            return

        for fill in fills:
            ticker = fill.get("ticker", "")
            if ticker not in ticker_to_oid:
                continue

            oid = ticker_to_oid[ticker]
            p = pending[oid]
            entry_cents = fill.get("yes_price", self.target_bid_cents)
            count = fill.get("count", self.order_size_contracts)

            pos = OpenPosition(
                ticker=ticker,
                event_ticker=p.intent.condition_id,
                entry_price_cents=entry_cents,
                count=count,
                filled_at=time.time(),
            )
            self.positions[ticker] = pos
            self.total_fills += 1

            # Remove from manager's pending
            await self.manager.cancel(oid)

            logger.info(
                "kalshi_fill_detected",
                ticker=ticker,
                price_cents=pos.entry_price_cents,
                count=pos.count,
            )

    async def _check_settlements(self) -> None:
        """Check if any held positions have settled."""
        settled_tickers: list[str] = []

        for ticker, pos in self.positions.items():
            try:
                data = await self.executor.api_get(f"/markets/{ticker}")
                if not data:
                    continue
                mkt = data.get("market", data)
                status = mkt.get("status", "")
                result = mkt.get("result", "")

                if status in ("settled", "finalized") and result:
                    won = result == "yes"
                    settlement_price = 1.0 if won else 0.0

                    await self.manager.settle(
                        condition_id=pos.event_ticker,
                        outcome="yes",
                        settlement_price=settlement_price,
                        won=won,
                    )
                    settled_tickers.append(ticker)

            except Exception as exc:
                logger.debug("kalshi_settlement_check_error", ticker=ticker, error=str(exc))

        for t in settled_tickers:
            del self.positions[t]

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    async def status_loop(self) -> None:
        while True:
            await asyncio.sleep(60.0)
            exposure = sum(
                p.count * p.entry_price_cents / 100.0 for p in self.positions.values()
            )
            stats = self.manager.get_stats()
            total_fills = self.total_fills
            wins = stats["wins"]
            losses = stats["losses"]
            pnl = stats["total_pnl"]
            winrate = wins / total_fills * 100 if total_fills > 0 else 0
            print(
                f"[{time.strftime('%H:%M:%S')}] "
                f"events={len(self.known_events)} "
                f"pending={stats['pending_orders']} "
                f"positions={len(self.positions)} "
                f"fills={total_fills} "
                f"record={wins}W-{losses}L "
                f"winrate={winrate:.1f}% "
                f"pnl=${pnl:+.2f} "
                f"exposure=${exposure:.0f}"
            )

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    async def run(self) -> None:
        try:
            logger.info(
                "kalshi_td_maker_started",
                paper=self.paper_mode,
                target_bid_cents=self.target_bid_cents,
                symbols=self.symbols,
                order_size=self.order_size_contracts,
                max_exposure=self.max_total_exposure_usd,
            )

            await asyncio.gather(
                self.scan_loop(),
                self.fill_check_loop(),
                self.status_loop(),
            )
        finally:
            await self.executor.close()
            await self.manager.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Passive time-decay maker for Kalshi crypto bracket markets"
    )
    p.add_argument("--paper", action="store_true", default=True)
    p.add_argument("--live", action="store_true", default=False)
    p.add_argument("--demo", action="store_true", default=False, help="Use Kalshi demo/sandbox API")
    p.add_argument("--target-bid", type=int, default=75, help="Target bid in cents (default: 75)")
    p.add_argument("--min-bid", type=int, default=65, help="Min acceptable bid (cents)")
    p.add_argument("--max-bid", type=int, default=85, help="Max acceptable bid (cents)")
    p.add_argument("--order-size", type=int, default=10, help="Contracts per order")
    p.add_argument("--max-exposure", type=float, default=200.0, help="Max total exposure USD")
    p.add_argument("--scan-interval", type=float, default=120.0, help="Scan interval seconds")
    p.add_argument("--symbols", type=str, default="KXBTCD,KXETHD", help="Kalshi event prefixes")
    p.add_argument("--api-key-id", type=str, default="", help="Kalshi API key ID")
    p.add_argument("--private-key-path", type=str, default="", help="Path to RSA private key PEM")
    return p


async def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    paper_mode = not args.live
    api_base = KALSHI_DEMO_API_BASE if args.demo else KALSHI_API_BASE
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]

    # Load private key if provided
    private_key_pem = ""
    key_path = args.private_key_path or getattr(settings, "KALSHI_PRIVATE_KEY_PATH", "")
    if key_path and Path(key_path).exists():
        private_key_pem = Path(key_path).read_text()

    api_key_id = args.api_key_id or getattr(settings, "KALSHI_API_KEY_ID", "")

    executor = KalshiExecutor(
        api_base=api_base,
        api_key_id=api_key_id,
        private_key_pem=private_key_pem,
    )

    manager = TradeManager(
        executor=executor,
        strategy="KalshiTDMaker",
        paper=paper_mode,
        db_url=settings.DATABASE_URL,
        event_type=KALSHI_TD_MAKER_EVENT_TYPE,
    )

    maker = KalshiTDMaker(
        executor=executor,
        manager=manager,
        target_bid_cents=args.target_bid,
        min_bid_cents=args.min_bid,
        max_bid_cents=args.max_bid,
        order_size_contracts=args.order_size,
        max_total_exposure_usd=args.max_exposure,
        paper_mode=paper_mode,
        scan_interval=args.scan_interval,
        symbols=symbols,
    )

    print(f"=== Kalshi TD Maker {'(PAPER)' if paper_mode else '(LIVE)'} ===")
    print(f"  API:         {'demo' if args.demo else 'production'}")
    print(f"  Symbols:     {', '.join(symbols)}")
    print(f"  Target bid:  {args.target_bid}c")
    print(f"  Range:       {args.min_bid}c - {args.max_bid}c")
    print(f"  Order size:  {args.order_size} contracts")
    print(f"  Max exposure: ${args.max_exposure}")
    print(f"  Strategy:    Find strike with YES ~{args.target_bid}c, bid GTC")
    print()

    await maker.run()


if __name__ == "__main__":
    asyncio.run(main())
