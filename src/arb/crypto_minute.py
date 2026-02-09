"""Crypto 15-Minute Binary Market Strategies.

Two complementary strategies on Polymarket 15-minute crypto binary markets:

- Time Decay: Buy the expensive side (>88c) when spot-vs-threshold gap is large
  enough that the current state should hold through expiration.
- Long Vol: Buy the cheap side (<15c) when the gap is small enough that a
  reversal is realistic before expiration.

Both strategies are symmetric (work for Up or Down, YES or NO).

Data sources:
- Polymarket Gamma API: discover markets, get outcome prices, resolve trades
- Binance public REST (no auth): spot price for BTC/ETH every few seconds
"""

import asyncio
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import structlog

from config.settings import settings

logger = structlog.get_logger()

# Binance symbol -> Polymarket slug prefix
SYMBOL_TO_SLUG: dict[str, str] = {
    "BTCUSDT": "btc",
    "ETHUSDT": "eth",
}


@dataclass
class MinuteMarket:
    """An active 15-minute crypto market discovered via Gamma API."""

    condition_id: str
    slug: str
    symbol: str
    event_start: float  # unix timestamp: start of the 15-min window
    end_time: float  # unix timestamp: end of the 15-min window
    token_ids: dict[str, str] = field(default_factory=dict)  # {"Up": "...", "Down": "..."}
    outcome_prices: dict[str, float] = field(default_factory=dict)  # {"Up": 0.92, "Down": 0.08}


@dataclass
class MinuteOpportunity:
    """A detected entry opportunity on a 15-min market."""

    market: MinuteMarket
    strategy: str  # "time_decay" or "long_vol"
    side: str  # "Up" or "Down"
    entry_price: float
    spot_price: float
    gap_pct: float
    time_remaining_s: int
    potential_profit: float  # 1 - entry_price


@dataclass
class PaperTrade:
    """A paper trade record with tags for analysis."""

    id: str = ""
    timestamp: str = ""
    strategy: str = ""  # "time_decay" | "long_vol"
    symbol: str = ""
    market_slug: str = ""
    side: str = ""  # "Up" | "Down"
    entry_price: float = 0.0
    spot_at_entry: float = 0.0
    gap_pct: float = 0.0
    gap_bucket: str = ""  # "small" | "medium" | "large"
    time_remaining_s: int = 0
    time_bucket: str = ""  # "2-3min" | "3-4min" | "4-5min"
    size_usd: float = 0.0
    resolved: bool = False
    won: bool = False
    pnl_usd: float = 0.0
    spot_at_resolution: float = 0.0
    resolution_time: str = ""

    @staticmethod
    def gap_to_bucket(gap_pct: float) -> str:
        if gap_pct < 0.2:
            return "small"
        elif gap_pct < 0.5:
            return "medium"
        return "large"

    @staticmethod
    def time_to_bucket(seconds: int) -> str:
        minutes = seconds / 60
        if minutes < 3:
            return "2-3min"
        elif minutes < 4:
            return "3-4min"
        return "4-5min"


class BinanceSpotPoller:
    """Simple HTTP poller for Binance spot prices (no auth required)."""

    def __init__(self, symbols: list[str], base_url: str = ""):
        self._symbols = symbols
        self._base_url = base_url or settings.CRYPTO_MINUTE_BINANCE_URL
        self._prices: dict[str, float] = {}
        self._last_poll: float = 0.0

    @property
    def prices(self) -> dict[str, float]:
        return dict(self._prices)

    async def poll(self) -> dict[str, float]:
        """Fetch current spot prices from Binance public API."""
        import aiohttp

        try:
            async with aiohttp.ClientSession() as session:
                for symbol in self._symbols:
                    url = f"{self._base_url}?symbol={symbol}"
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            self._prices[symbol] = float(data["price"])
            self._last_poll = time.time()
        except Exception as e:
            logger.warning("binance_poll_error", error=str(e))

        return dict(self._prices)


class MarketScanner:
    """Discovers and tracks active 15-minute crypto markets via Gamma API."""

    def __init__(self, symbols: list[str], gamma_url: str = ""):
        self._symbols = symbols
        self._gamma_url = gamma_url or settings.CRYPTO_MINUTE_GAMMA_URL
        self._markets: dict[str, MinuteMarket] = {}  # slug -> market
        self._last_sync: float = 0.0

    @property
    def markets(self) -> dict[str, MinuteMarket]:
        return dict(self._markets)

    async def sync(self) -> list[MinuteMarket]:
        """Fetch currently active 15-min markets from Gamma API."""
        import aiohttp

        now = time.time()
        discovered: list[MinuteMarket] = []

        try:
            async with aiohttp.ClientSession() as session:
                for symbol in self._symbols:
                    slug_prefix = SYMBOL_TO_SLUG.get(symbol)
                    if not slug_prefix:
                        continue

                    # Current and next 15-min slot
                    current_slot = int(now // 900) * 900
                    for slot_ts in [current_slot, current_slot + 900]:
                        slug = f"{slug_prefix}-updown-15m-{slot_ts}"
                        if slug in self._markets:
                            continue

                        url = f"{self._gamma_url}/events?slug={slug}"
                        async with session.get(
                            url,
                            headers={"User-Agent": "Mozilla/5.0"},
                            timeout=aiohttp.ClientTimeout(total=10),
                        ) as resp:
                            if resp.status != 200:
                                continue
                            events = await resp.json()
                            if not events:
                                continue

                            event = events[0]
                            markets_data = event.get("markets", [])
                            if not markets_data:
                                continue

                            mkt = markets_data[0]
                            market = self._parse_market(mkt, symbol, slug)
                            if market:
                                self._markets[slug] = market
                                discovered.append(market)

        except Exception as e:
            logger.warning("gamma_sync_error", error=str(e))

        self._last_sync = now
        self._cleanup_expired(now)
        return discovered

    def _parse_market(
        self, mkt: dict[str, Any], symbol: str, slug: str
    ) -> Optional[MinuteMarket]:
        """Parse a Gamma API market response into a MinuteMarket."""
        try:
            condition_id = mkt.get("conditionId", "")
            outcomes = json.loads(mkt.get("outcomes", "[]"))
            outcome_prices_raw = json.loads(mkt.get("outcomePrices", "[]"))
            token_ids_raw = json.loads(mkt.get("clobTokenIds", "[]"))

            if len(outcomes) < 2 or len(token_ids_raw) < 2:
                return None

            token_ids = {}
            outcome_prices = {}
            for i, outcome in enumerate(outcomes):
                token_ids[outcome] = token_ids_raw[i] if i < len(token_ids_raw) else ""
                outcome_prices[outcome] = (
                    float(outcome_prices_raw[i]) if i < len(outcome_prices_raw) else 0.0
                )

            # Parse event start time
            event_start_str = mkt.get("eventStartTime") or mkt.get("startDate", "")
            end_str = mkt.get("endDate", "")

            event_start = self._parse_iso(event_start_str)
            end_time = self._parse_iso(end_str)

            if not event_start or not end_time:
                return None

            return MinuteMarket(
                condition_id=condition_id,
                slug=slug,
                symbol=symbol,
                event_start=event_start,
                end_time=end_time,
                token_ids=token_ids,
                outcome_prices=outcome_prices,
            )
        except Exception as e:
            logger.warning("parse_market_error", slug=slug, error=str(e))
            return None

    @staticmethod
    def _parse_iso(iso_str: str) -> Optional[float]:
        if not iso_str:
            return None
        try:
            # Handle both with and without Z suffix
            iso_str = iso_str.replace("Z", "+00:00")
            dt = datetime.fromisoformat(iso_str)
            return dt.timestamp()
        except ValueError:
            return None

    def _cleanup_expired(self, now: float) -> None:
        """Remove markets that have already ended."""
        expired = [slug for slug, m in self._markets.items() if m.end_time < now - 60]
        for slug in expired:
            del self._markets[slug]

    async def refresh_prices(self, slug: str) -> Optional[MinuteMarket]:
        """Re-fetch outcome prices for a specific market (for resolution)."""
        import aiohttp

        market = self._markets.get(slug)
        if not market:
            return None

        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self._gamma_url}/events?slug={slug}"
                async with session.get(
                    url,
                    headers={"User-Agent": "Mozilla/5.0"},
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status != 200:
                        return market
                    events = await resp.json()
                    if not events:
                        return market

                    mkt = events[0].get("markets", [{}])[0]
                    outcome_prices_raw = json.loads(mkt.get("outcomePrices", "[]"))
                    outcomes = json.loads(mkt.get("outcomes", "[]"))

                    for i, outcome in enumerate(outcomes):
                        if i < len(outcome_prices_raw):
                            market.outcome_prices[outcome] = float(outcome_prices_raw[i])

        except Exception as e:
            logger.warning("refresh_prices_error", slug=slug, error=str(e))

        return market


class CryptoMinuteEngine:
    """Main engine for 15-minute crypto binary market strategies."""

    def __init__(
        self,
        poller: Optional[BinanceSpotPoller] = None,
        scanner: Optional[MarketScanner] = None,
    ):
        symbols = settings.CRYPTO_MINUTE_SYMBOLS.split(",")
        self.poller = poller or BinanceSpotPoller(symbols)
        self.scanner = scanner or MarketScanner(symbols)

        # Strategy thresholds
        self.td_threshold = settings.CRYPTO_MINUTE_TD_THRESHOLD
        self.td_min_gap_pct = settings.CRYPTO_MINUTE_TD_MIN_GAP_PCT
        self.lv_threshold = settings.CRYPTO_MINUTE_LV_THRESHOLD
        self.lv_max_gap_pct = settings.CRYPTO_MINUTE_LV_MAX_GAP_PCT

        # Entry window
        self.min_entry_time = settings.CRYPTO_MINUTE_MIN_ENTRY_TIME
        self.max_entry_time = settings.CRYPTO_MINUTE_MAX_ENTRY_TIME

        # Paper trading state
        self.paper_size = settings.CRYPTO_MINUTE_PAPER_SIZE_USD
        self.paper_file = Path(settings.CRYPTO_MINUTE_PAPER_FILE)
        self._open_trades: list[PaperTrade] = []
        self._stats: dict[str, dict[str, float]] = {
            "time_decay": {"trades": 0, "wins": 0, "pnl": 0.0},
            "long_vol": {"trades": 0, "wins": 0, "pnl": 0.0},
        }

        # Track which markets we already entered to avoid duplicates
        self._entered_markets: dict[str, set[str]] = {}  # slug -> set of strategies

        # Spot prices at event start (for gap calculation)
        self._start_prices: dict[str, float] = {}  # slug -> spot price at event_start

    async def scan_once(self) -> list[MinuteOpportunity]:
        """Run a single scan cycle: poll prices, discover markets, evaluate."""
        # 1. Poll Binance spot prices
        spot_prices = await self.poller.poll()
        if not spot_prices:
            return []

        # 2. Discover / refresh markets
        await self.scanner.sync()

        now = time.time()
        opportunities: list[MinuteOpportunity] = []

        for slug, market in self.scanner.markets.items():
            # Record spot price at event start for gap calculation
            if slug not in self._start_prices:
                spot = spot_prices.get(market.symbol)
                if spot:
                    self._start_prices[slug] = spot

            # Check time window
            time_remaining = market.end_time - now
            if time_remaining < self.min_entry_time or time_remaining > self.max_entry_time:
                continue

            spot = spot_prices.get(market.symbol)
            start_spot = self._start_prices.get(slug)
            if not spot or not start_spot or start_spot == 0:
                continue

            # Calculate gap: how far is spot from the threshold (start price)
            gap_pct = abs(spot - start_spot) / start_spot * 100

            # Find cheap and expensive sides
            for side, price in market.outcome_prices.items():
                # Time Decay: buy expensive side
                if price >= self.td_threshold:
                    entered = self._entered_markets.get(slug, set())
                    if "time_decay" not in entered:
                        if gap_pct >= self.td_min_gap_pct:
                            opp = MinuteOpportunity(
                                market=market,
                                strategy="time_decay",
                                side=side,
                                entry_price=price,
                                spot_price=spot,
                                gap_pct=gap_pct,
                                time_remaining_s=int(time_remaining),
                                potential_profit=1.0 - price,
                            )
                            opportunities.append(opp)

                # Long Vol: buy cheap side
                if price <= self.lv_threshold:
                    entered = self._entered_markets.get(slug, set())
                    if "long_vol" not in entered:
                        if gap_pct <= self.lv_max_gap_pct:
                            opp = MinuteOpportunity(
                                market=market,
                                strategy="long_vol",
                                side=side,
                                entry_price=price,
                                spot_price=spot,
                                gap_pct=gap_pct,
                                time_remaining_s=int(time_remaining),
                                potential_profit=1.0 - price,
                            )
                            opportunities.append(opp)

        return opportunities

    def enter_paper_trade(self, opp: MinuteOpportunity) -> PaperTrade:
        """Record a paper trade entry."""
        trade = PaperTrade(
            id=str(uuid.uuid4())[:8],
            timestamp=datetime.now(timezone.utc).isoformat(),
            strategy=opp.strategy,
            symbol=opp.market.symbol,
            market_slug=opp.market.slug,
            side=opp.side,
            entry_price=opp.entry_price,
            spot_at_entry=opp.spot_price,
            gap_pct=round(opp.gap_pct, 4),
            gap_bucket=PaperTrade.gap_to_bucket(opp.gap_pct),
            time_remaining_s=opp.time_remaining_s,
            time_bucket=PaperTrade.time_to_bucket(opp.time_remaining_s),
            size_usd=self.paper_size,
        )

        self._open_trades.append(trade)

        # Mark market+strategy as entered
        if opp.market.slug not in self._entered_markets:
            self._entered_markets[opp.market.slug] = set()
        self._entered_markets[opp.market.slug].add(opp.strategy)

        logger.info(
            "paper_trade_entered",
            strategy=trade.strategy,
            symbol=trade.symbol,
            side=trade.side,
            entry_price=trade.entry_price,
            gap_pct=trade.gap_pct,
            time_remaining=trade.time_remaining_s,
        )
        return trade

    async def resolve_expired_trades(self) -> list[PaperTrade]:
        """Check and resolve any trades whose markets have expired."""
        now = time.time()
        spot_prices = self.poller.prices
        resolved: list[PaperTrade] = []

        still_open: list[PaperTrade] = []
        for trade in self._open_trades:
            market = self.scanner.markets.get(trade.market_slug)

            # Market already expired or not found
            if not market or market.end_time <= now:
                # Refresh final prices
                if market:
                    await self.scanner.refresh_prices(trade.market_slug)
                    market = self.scanner.markets.get(trade.market_slug)

                # Resolve
                final_price = 0.0
                if market:
                    final_price = market.outcome_prices.get(trade.side, 0.0)

                # Won if final price > entry price (for both strategies)
                # At expiration, prices converge to 0 or 1
                won = final_price > trade.entry_price
                pnl_per_unit = final_price - trade.entry_price
                trade.won = won
                trade.pnl_usd = round(pnl_per_unit * (trade.size_usd / trade.entry_price), 2)
                trade.resolved = True
                trade.spot_at_resolution = spot_prices.get(trade.symbol, 0.0)
                trade.resolution_time = datetime.now(timezone.utc).isoformat()

                # Update stats
                self._stats[trade.strategy]["trades"] += 1
                if won:
                    self._stats[trade.strategy]["wins"] += 1
                self._stats[trade.strategy]["pnl"] += trade.pnl_usd

                self._save_trade(trade)
                resolved.append(trade)

                logger.info(
                    "paper_trade_resolved",
                    strategy=trade.strategy,
                    symbol=trade.symbol,
                    side=trade.side,
                    won=won,
                    pnl=trade.pnl_usd,
                    entry=trade.entry_price,
                    final=round(final_price, 3),
                )
            else:
                still_open.append(trade)

        self._open_trades = still_open
        return resolved

    def _save_trade(self, trade: PaperTrade) -> None:
        """Append a resolved trade to the JSONL file."""
        self.paper_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.paper_file, "a") as f:
            f.write(json.dumps(asdict(trade)) + "\n")

    def get_stats(self) -> dict[str, Any]:
        """Return current paper trading stats."""
        stats = {}
        for strategy, data in self._stats.items():
            total = data["trades"]
            wins = data["wins"]
            stats[strategy] = {
                "trades": total,
                "wins": wins,
                "winrate": round(wins / total * 100, 1) if total > 0 else 0.0,
                "pnl_usd": round(data["pnl"], 2),
                "open": len([t for t in self._open_trades if t.strategy == strategy]),
            }
        return stats

    async def run(self) -> None:
        """Main loop: scan, enter trades, resolve expired ones."""
        scan_interval = settings.CRYPTO_MINUTE_SCAN_INTERVAL
        logger.info(
            "crypto_minute_engine_started",
            symbols=settings.CRYPTO_MINUTE_SYMBOLS,
            scan_interval=scan_interval,
            td_threshold=self.td_threshold,
            lv_threshold=self.lv_threshold,
        )

        while True:
            try:
                # Scan for opportunities
                opportunities = await self.scan_once()
                for opp in opportunities:
                    self.enter_paper_trade(opp)

                # Resolve expired trades
                await self.resolve_expired_trades()

                # Log stats periodically
                stats = self.get_stats()
                td = stats.get("time_decay", {})
                lv = stats.get("long_vol", {})
                logger.info(
                    "crypto_minute_stats",
                    td_trades=td.get("trades", 0),
                    td_winrate=td.get("winrate", 0),
                    td_pnl=td.get("pnl_usd", 0),
                    lv_trades=lv.get("trades", 0),
                    lv_winrate=lv.get("winrate", 0),
                    lv_pnl=lv.get("pnl_usd", 0),
                    open_trades=len(self._open_trades),
                )

            except Exception as e:
                logger.error("crypto_minute_scan_error", error=str(e))

            await asyncio.sleep(scan_interval)
