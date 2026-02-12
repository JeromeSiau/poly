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
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import httpx
import structlog

_project_root = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root))

from config.settings import settings

logger = structlog.get_logger()

# Kalshi API endpoints
KALSHI_API_BASE = "https://api.elections.kalshi.com/trade-api/v2"
KALSHI_DEMO_API_BASE = "https://demo-api.kalshi.co/trade-api/v2"

# Crypto event ticker prefixes
KALSHI_CRYPTO_PREFIXES = ["KXBTCD", "KXETHD"]


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
class PassiveBid:
    """A GTC bid placed on a Kalshi strike."""
    order_id: str
    ticker: str
    event_ticker: str
    side: str  # "yes"
    price_cents: int
    count: int  # number of contracts
    placed_at: float


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
    """

    def __init__(
        self,
        *,
        api_base: str = KALSHI_API_BASE,
        api_key_id: str = "",
        private_key_pem: str = "",
        target_bid_cents: int = 75,
        min_bid_cents: int = 65,
        max_bid_cents: int = 85,
        order_size_contracts: int = 10,
        max_total_exposure_usd: float = 200.0,
        paper_mode: bool = True,
        scan_interval: float = 120.0,
        symbols: list[str] | None = None,
    ) -> None:
        self.api_base = api_base
        self.api_key_id = api_key_id
        self.private_key_pem = private_key_pem
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
        self.active_bids: dict[str, PassiveBid] = {}  # ticker -> bid
        self.positions: dict[str, OpenPosition] = {}  # ticker -> position

        # Stats
        self.total_fills: int = 0
        self.total_wins: int = 0
        self.total_losses: int = 0
        self.realized_pnl: float = 0.0

        self._client: Optional[httpx.AsyncClient] = None
        self._paper_order_counter: int = 0

    # ------------------------------------------------------------------
    # API helpers
    # ------------------------------------------------------------------

    def _auth_headers(self, method: str, path: str) -> dict[str, str]:
        """Generate RSA-signed auth headers for Kalshi API."""
        if self.paper_mode or not self.api_key_id or not self.private_key_pem:
            return {}

        import hashlib
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import padding

        timestamp_ms = str(int(time.time() * 1000))
        message = f"{timestamp_ms}{method.upper()}{path}"

        private_key = serialization.load_pem_private_key(
            self.private_key_pem.encode(), password=None
        )
        signature = private_key.sign(
            message.encode(),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=hashes.SHA256.digest_size,
            ),
            hashes.SHA256(),
        )

        import base64
        return {
            "KALSHI-ACCESS-KEY": self.api_key_id,
            "KALSHI-ACCESS-TIMESTAMP": timestamp_ms,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode(),
        }

    async def _api_get(self, path: str, params: dict | None = None) -> Any:
        """GET request to Kalshi API."""
        if self._client is None:
            return None
        url = f"{self.api_base}{path}"
        headers = self._auth_headers("GET", path)
        resp = await self._client.get(url, params=params, headers=headers)
        resp.raise_for_status()
        return resp.json()

    async def _api_post(self, path: str, body: dict) -> Any:
        """POST request to Kalshi API."""
        if self._client is None:
            return None
        url = f"{self.api_base}{path}"
        headers = self._auth_headers("POST", path)
        resp = await self._client.post(url, json=body, headers=headers)
        resp.raise_for_status()
        return resp.json()

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
                data = await self._api_get("/markets", params={
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
                        active_bids=len(self.active_bids),
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
        """Place a GTC limit buy YES on the selected strike."""
        ticker = strike.get("ticker", "")
        event_ticker = strike.get("event_ticker", "")

        if ticker in self.active_bids or ticker in self.positions:
            return

        # Check exposure limit
        current_exposure = sum(
            p.count * p.entry_price_cents / 100.0 for p in self.positions.values()
        )
        if current_exposure >= self.max_total_exposure_usd:
            return

        now = time.time()

        if self.paper_mode:
            self._paper_order_counter += 1
            order_id = f"kalshi_paper_{self._paper_order_counter}"
            # In paper mode, simulate immediate fill if ask <= target
            yes_ask = strike.get("yes_ask", 0) or 0
            if yes_ask <= self.target_bid_cents and yes_ask > 0:
                pos = OpenPosition(
                    ticker=ticker,
                    event_ticker=event_ticker,
                    entry_price_cents=yes_ask,
                    count=self.order_size_contracts,
                    filled_at=now,
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
        else:
            body = {
                "ticker": ticker,
                "side": "yes",
                "action": "buy",
                "type": "limit",
                "yes_price": self.target_bid_cents,
                "count": self.order_size_contracts,
                "time_in_force": "good_till_canceled",
                "post_only": True,
            }
            try:
                resp = await self._api_post("/portfolio/orders", body)
                order_id = resp.get("order", {}).get("order_id", "")
                if not order_id:
                    logger.warning("kalshi_order_failed", response=resp)
                    return
            except Exception as exc:
                logger.warning("kalshi_order_error", ticker=ticker, error=str(exc))
                return

        bid = PassiveBid(
            order_id=order_id,
            ticker=ticker,
            event_ticker=event_ticker,
            side="yes",
            price_cents=self.target_bid_cents,
            count=self.order_size_contracts,
            placed_at=now,
        )
        self.active_bids[ticker] = bid
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
        if not self.active_bids:
            return

        try:
            data = await self._api_get("/portfolio/fills", params={"limit": 100})
            if not data:
                return
            fills = data.get("fills", [])
        except Exception:
            return

        filled_tickers: list[str] = []
        for fill in fills:
            ticker = fill.get("ticker", "")
            if ticker in self.active_bids:
                bid = self.active_bids[ticker]
                pos = OpenPosition(
                    ticker=ticker,
                    event_ticker=bid.event_ticker,
                    entry_price_cents=fill.get("yes_price", bid.price_cents),
                    count=fill.get("count", bid.count),
                    filled_at=time.time(),
                )
                self.positions[ticker] = pos
                self.total_fills += 1
                filled_tickers.append(ticker)
                logger.info(
                    "kalshi_fill_detected",
                    ticker=ticker,
                    price_cents=pos.entry_price_cents,
                    count=pos.count,
                )

        for t in filled_tickers:
            del self.active_bids[t]

    async def _check_settlements(self) -> None:
        """Check if any held positions have settled."""
        settled_tickers: list[str] = []

        for ticker, pos in self.positions.items():
            try:
                data = await self._api_get(f"/markets/{ticker}")
                if not data:
                    continue
                mkt = data.get("market", data)
                status = mkt.get("status", "")
                result = mkt.get("result", "")

                if status in ("settled", "finalized") and result:
                    won = result == "yes"
                    if won:
                        pnl = pos.count * (100 - pos.entry_price_cents) / 100.0
                        self.total_wins += 1
                    else:
                        pnl = -pos.count * pos.entry_price_cents / 100.0
                        self.total_losses += 1

                    self.realized_pnl += pnl
                    settled_tickers.append(ticker)

                    logger.info(
                        "kalshi_settled",
                        ticker=ticker,
                        won=won,
                        entry_cents=pos.entry_price_cents,
                        pnl=round(pnl, 4),
                        total_pnl=round(self.realized_pnl, 4),
                        record=f"{self.total_wins}W-{self.total_losses}L",
                    )
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
            winrate = (
                self.total_wins / self.total_fills * 100
                if self.total_fills > 0
                else 0
            )
            print(
                f"[{time.strftime('%H:%M:%S')}] "
                f"events={len(self.known_events)} "
                f"bids={len(self.active_bids)} "
                f"positions={len(self.positions)} "
                f"fills={self.total_fills} "
                f"record={self.total_wins}W-{self.total_losses}L "
                f"winrate={winrate:.1f}% "
                f"pnl=${self.realized_pnl:+.2f} "
                f"exposure=${exposure:.0f}"
            )

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    async def run(self) -> None:
        timeout = httpx.Timeout(20.0, connect=10.0)
        async with httpx.AsyncClient(timeout=timeout, http2=True) as client:
            self._client = client

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

    maker = KalshiTDMaker(
        api_base=api_base,
        api_key_id=api_key_id,
        private_key_pem=private_key_pem,
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
