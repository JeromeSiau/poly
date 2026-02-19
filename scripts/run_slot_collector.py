"""Passive slot data collector for crypto binary markets (5m / 15m).

Snapshots Polymarket book + Chainlink prices at regular intervals for each
active slot, then resolves outcome after the slot expires.  Data stored in
MySQL for downstream ML training.

Usage:
    python scripts/run_slot_collector.py --symbols BTC,ETH,SOL,XRP
    python scripts/run_slot_collector.py --symbols BTC --slot-duration 5m --snapshot-interval 15
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from datetime import datetime, timezone
from typing import Any, Optional

import httpx
import structlog

try:
    import uvloop
except ImportError:
    uvloop = None

from src.utils.logging import configure_logging

configure_logging()

from config.settings import settings
from src.db.models import Base as SlotBase, SlotSnapshot, SlotResolution
from src.feeds.chainlink import ChainlinkFeed
from src.feeds.polymarket import PolymarketFeed
from src.utils.crypto_markets import (
    CRYPTO_SYMBOL_TO_SLUG,
    SLUG_TO_CHAINLINK,
    fetch_crypto_markets,
)
from src.utils.parsing import _first_event_slug, parse_json_list

from sqlalchemy import select, update as sa_update
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

logger = structlog.get_logger()

GAMMA_URL = "https://gamma-api.polymarket.com"
CLOB_URL = "https://clob.polymarket.com"


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _slug_to_symbol(slug: str) -> Optional[str]:
    """Extract short symbol from slug like 'btc-updown-15m-1771079400' → 'BTC'."""
    parts = slug.split("-")
    if len(parts) >= 4:
        return parts[0].upper()
    return None


DURATION_MAP = {"5m": 300, "15m": 900}


def _slug_to_slot_ts(slug: str) -> Optional[int]:
    """Extract slot timestamp from slug."""
    parts = slug.split("-")
    if len(parts) >= 4 and parts[-1].isdigit():
        return int(parts[-1])
    return None


def _slug_to_chainlink_sym(slug: str) -> Optional[str]:
    """Extract chainlink symbol from slug."""
    parts = slug.split("-")
    if parts:
        return SLUG_TO_CHAINLINK.get(parts[0].lower())
    return None


# ------------------------------------------------------------------
# Active slot tracker
# ------------------------------------------------------------------

class ActiveSlot:
    """State for a single active slot (5m or 15m)."""

    __slots__ = (
        "symbol", "slot_ts", "slot_duration", "condition_id", "chainlink_sym",
        "ref_price", "outcomes", "token_ids", "market_volume_usd",
    )

    def __init__(
        self,
        symbol: str,
        slot_ts: int,
        slot_duration: int,
        condition_id: str,
        chainlink_sym: str,
        ref_price: Optional[float],
        outcomes: list[str],
        token_ids: list[str],
        market_volume_usd: Optional[float] = None,
    ) -> None:
        self.symbol = symbol
        self.slot_ts = slot_ts
        self.slot_duration = slot_duration
        self.condition_id = condition_id
        self.chainlink_sym = chainlink_sym
        self.ref_price = ref_price
        self.outcomes = outcomes
        self.token_ids = token_ids
        self.market_volume_usd = market_volume_usd


# ------------------------------------------------------------------
# Collector
# ------------------------------------------------------------------

class SlotCollector:
    """Passive data collector for crypto slots (5m or 15m)."""

    def __init__(
        self,
        *,
        polymarket: PolymarketFeed,
        chainlink: ChainlinkFeed,
        symbols: list[str],
        db_url: str,
        slot_duration: int = 900,
        discovery_interval: float = 60.0,
        snapshot_interval: float = 30.0,
        resolution_interval: float = 60.0,
    ) -> None:
        self.polymarket = polymarket
        self.chainlink = chainlink
        self.symbols = symbols
        self.slot_duration = slot_duration
        self.discovery_interval = discovery_interval
        self.snapshot_interval = snapshot_interval
        self.resolution_interval = resolution_interval

        # Async engine for slot data persistence
        self._engine = create_async_engine(
            db_url,
            pool_size=5,
            max_overflow=5,
            pool_pre_ping=True,
            echo=False,
        )
        self._session_factory = async_sessionmaker(
            bind=self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

        # State
        self._active_slots: dict[str, ActiveSlot] = {}  # cid -> ActiveSlot
        self._known_cids: set[str] = set()
        self._http_client: Optional[httpx.AsyncClient] = None
        # Track previous resolution per symbol for prev_resolved_up
        self._prev_resolved: dict[str, bool] = {}  # symbol -> last resolved_up
        # Write queue for fire-and-forget DB writes
        self._write_queue: asyncio.Queue = asyncio.Queue(maxsize=2000)

    # ------------------------------------------------------------------
    # DB helpers
    # ------------------------------------------------------------------

    async def _init_tables(self) -> None:
        """Create tables if they don't exist."""
        async with self._engine.begin() as conn:
            await conn.run_sync(SlotBase.metadata.create_all)
        logger.info("slot_collector_tables_ready")

    async def _db_writer_loop(self) -> None:
        """Drain the write queue and batch-insert into MySQL."""
        while True:
            items: list[Any] = []
            # Wait for at least one item
            item = await self._write_queue.get()
            items.append(item)
            # Drain remaining without blocking
            while not self._write_queue.empty() and len(items) < 100:
                try:
                    items.append(self._write_queue.get_nowait())
                except asyncio.QueueEmpty:
                    break

            try:
                async with self._session_factory() as session:
                    session.add_all(items)
                    await session.commit()
            except Exception as exc:
                logger.warning("slot_db_write_failed", count=len(items), error=str(exc)[:120])

    def _enqueue(self, obj: Any) -> None:
        """Fire-and-forget DB write."""
        try:
            self._write_queue.put_nowait(obj)
        except asyncio.QueueFull:
            logger.warning("slot_write_queue_full")

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    async def _discovery_loop(self) -> None:
        while True:
            try:
                await self._discover()
            except Exception as exc:
                logger.error("slot_discovery_error", error=str(exc)[:120])
            await asyncio.sleep(self.discovery_interval)

    async def _discover(self) -> None:
        if not self._http_client:
            return

        raw_markets = await fetch_crypto_markets(
            client=self._http_client,
            symbols=self.symbols,
            slot_duration_sec=self.slot_duration,
        )

        new_count = 0
        for mkt in raw_markets:
            cid = str(mkt.get("conditionId", ""))
            if not cid:
                continue

            # Refresh volume on already-tracked slots (discovery runs every 60s)
            if cid in self._known_cids:
                if cid in self._active_slots:
                    vol_raw = mkt.get("volume")
                    if vol_raw is not None:
                        try:
                            self._active_slots[cid].market_volume_usd = float(vol_raw)
                        except (ValueError, TypeError):
                            pass
                continue

            outcomes = [str(o) for o in parse_json_list(mkt.get("outcomes", []))]
            clob_ids = [str(t) for t in parse_json_list(mkt.get("clobTokenIds", []))]
            if len(outcomes) < 2 or len(clob_ids) < 2:
                continue

            slug = _first_event_slug(mkt)
            symbol = _slug_to_symbol(slug)
            slot_ts = _slug_to_slot_ts(slug)
            chainlink_sym = _slug_to_chainlink_sym(slug)
            if not symbol or not slot_ts or not chainlink_sym:
                continue

            # Snapshot reference price
            ref_price = self.chainlink.get_price(chainlink_sym)

            vol_raw = mkt.get("volume")
            try:
                market_volume_usd = float(vol_raw) if vol_raw is not None else None
            except (ValueError, TypeError):
                market_volume_usd = None

            slot = ActiveSlot(
                symbol=symbol,
                slot_ts=slot_ts,
                slot_duration=self.slot_duration,
                condition_id=cid,
                chainlink_sym=chainlink_sym,
                ref_price=ref_price,
                outcomes=outcomes[:2],
                token_ids=clob_ids[:2],
                market_volume_usd=market_volume_usd,
            )
            self._active_slots[cid] = slot
            self._known_cids.add(cid)

            # Subscribe to book for this market
            token_map = {outcomes[i]: clob_ids[i] for i in range(min(2, len(outcomes)))}
            try:
                await self.polymarket.subscribe_market(cid, token_map=token_map, send=False)
            except Exception as exc:
                logger.warning("slot_subscribe_failed", cid=cid[:16], error=str(exc)[:60])

            # Insert slot_resolutions row
            prev = self._prev_resolved.get(symbol)
            self._enqueue(SlotResolution(
                symbol=symbol,
                slot_ts=slot_ts,
                slot_duration=self.slot_duration,
                condition_id=cid,
                prev_resolved_up=prev,
            ))

            new_count += 1

        if new_count:
            try:
                await self.polymarket.flush_subscriptions()
            except Exception as exc:
                logger.warning("slot_flush_failed", error=str(exc)[:60])

            logger.info(
                "slot_markets_discovered",
                new=new_count,
                active=len(self._active_slots),
            )

        # Prune expired slots (slot duration + 5 min grace)
        now = time.time()
        prune_after = self.slot_duration + 5 * 60
        expired = [
            cid for cid, s in self._active_slots.items()
            if now - s.slot_ts > prune_after
        ]
        for cid in expired:
            del self._active_slots[cid]

    # ------------------------------------------------------------------
    # Snapshots
    # ------------------------------------------------------------------

    async def _snapshot_loop(self) -> None:
        while True:
            try:
                self._take_snapshots()
            except Exception as exc:
                logger.error("slot_snapshot_error", error=str(exc)[:120])
            await asyncio.sleep(self.snapshot_interval)

    def _take_snapshots(self) -> None:
        now = time.time()
        dt = datetime.fromtimestamp(now, tz=timezone.utc)

        slot_max_min = self.slot_duration / 60 + 0.5

        for cid, slot in self._active_slots.items():
            minutes = (now - slot.slot_ts) / 60
            if minutes < 0 or minutes > slot_max_min:
                continue

            # Book levels
            bid_up, bid_sz_up, ask_up, ask_sz_up = (None, None, None, None)
            bid_dn, bid_sz_dn, ask_dn, ask_sz_dn = (None, None, None, None)

            if len(slot.outcomes) >= 2:
                bid_up, bid_sz_up, ask_up, ask_sz_up = self.polymarket.get_best_levels(
                    cid, slot.outcomes[0],  # Up
                )
                bid_dn, bid_sz_dn, ask_dn, ask_sz_dn = self.polymarket.get_best_levels(
                    cid, slot.outcomes[1],  # Down
                )

            # Chainlink
            current_price = self.chainlink.get_price(slot.chainlink_sym)
            dir_move = None
            abs_move = None
            if current_price and slot.ref_price:
                pct = (current_price - slot.ref_price) / slot.ref_price * 100
                dir_move = pct  # positive = Up direction
                abs_move = abs(pct)

            spread_up = None
            spread_dn = None
            if ask_up is not None and bid_up is not None:
                spread_up = ask_up - bid_up
            if ask_dn is not None and bid_dn is not None:
                spread_dn = ask_dn - bid_dn

            self._enqueue(SlotSnapshot(
                symbol=slot.symbol,
                slot_ts=slot.slot_ts,
                slot_duration=self.slot_duration,
                captured_at=now,
                minutes_into_slot=round(minutes, 2),
                bid_up=bid_up,
                ask_up=ask_up,
                bid_down=bid_dn,
                ask_down=ask_dn,
                bid_size_up=bid_sz_up,
                ask_size_up=ask_sz_up,
                bid_size_down=bid_sz_dn,
                ask_size_down=ask_sz_dn,
                spread_up=spread_up,
                spread_down=spread_dn,
                chainlink_price=current_price,
                dir_move_pct=round(dir_move, 4) if dir_move is not None else None,
                abs_move_pct=round(abs_move, 4) if abs_move is not None else None,
                market_volume_usd=slot.market_volume_usd,
                hour_utc=dt.hour,
                day_of_week=dt.weekday(),
            ))

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    async def _resolution_loop(self) -> None:
        while True:
            try:
                await self._resolve_pending()
            except Exception as exc:
                logger.error("slot_resolution_error", error=str(exc)[:120])
            await asyncio.sleep(self.resolution_interval)

    async def _resolve_pending(self) -> None:
        """Check unresolved slots and update resolved_up."""
        if not self._http_client:
            return

        now = time.time()

        # Query DB for unresolved slots (at least slot_duration old, give up after 2h)
        try:
            async with self._session_factory() as session:
                cutoff = now - self.slot_duration  # at least slot_duration old
                max_age = now - 120 * 60  # give up after 2 hours
                stmt = (
                    select(SlotResolution)
                    .where(SlotResolution.resolved_up.is_(None))
                    .where(SlotResolution.slot_ts < cutoff)
                    .where(SlotResolution.slot_ts > max_age)
                )
                result = await session.execute(stmt)
                pending = result.scalars().all()
        except Exception as exc:
            logger.warning("slot_resolution_query_failed", error=str(exc)[:120])
            return

        for row in pending:
            resolved = await self._query_resolution(row.condition_id, row.symbol)
            if resolved is None:
                continue

            # Update DB
            try:
                async with self._session_factory() as session:
                    stmt = (
                        sa_update(SlotResolution)
                        .where(SlotResolution.id == row.id)
                        .values(resolved_up=resolved, resolved_at=now)
                    )
                    await session.execute(stmt)
                    await session.commit()
            except Exception as exc:
                logger.warning("slot_resolution_update_failed", error=str(exc)[:60])
                continue

            # Track for prev_resolved_up + clean up tracking set
            self._prev_resolved[row.symbol] = resolved
            if row.condition_id:
                self._known_cids.discard(row.condition_id)

            logger.info(
                "slot_resolved",
                symbol=row.symbol,
                slot_ts=row.slot_ts,
                resolved_up=resolved,
            )

    async def _query_resolution(self, condition_id: str, symbol: str) -> Optional[bool]:
        """Query Gamma + CLOB for resolution. Returns True/False/None."""
        if not self._http_client or not condition_id:
            return None

        # CLOB API — query by condition_id, check both outcomes
        try:
            resp = await self._http_client.get(
                f"{CLOB_URL}/markets/{condition_id}",
                timeout=10.0,
            )
            if resp.status_code == 200:
                data = resp.json()
                tokens = data.get("tokens", [])
                for tok in tokens:
                    outcome = tok.get("outcome", "")
                    price_str = tok.get("price")
                    if price_str is None:
                        continue
                    price = float(price_str)
                    if outcome == "Up":
                        if price >= 0.9:
                            return True
                        if price <= 0.1:
                            return False
                    elif outcome == "Down":
                        if price >= 0.9:
                            return False
                        if price <= 0.1:
                            return True
        except Exception as exc:
            logger.warning(
                "slot_clob_resolution_failed",
                cid=condition_id[:16],
                error=str(exc)[:80],
            )

        return None

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    async def run(self) -> None:
        await self._init_tables()
        await self.polymarket.connect()
        await self.chainlink.connect()
        await asyncio.sleep(2)  # let Chainlink WS populate prices

        timeout = httpx.Timeout(20.0, connect=10.0)
        limits = httpx.Limits(max_connections=10, max_keepalive_connections=5)
        async with httpx.AsyncClient(timeout=timeout, limits=limits, http2=True) as client:
            self._http_client = client

            # Initial discovery
            await self._discover()

            logger.info(
                "slot_collector_started",
                symbols=self.symbols,
                slot_duration=self.slot_duration,
                active_slots=len(self._active_slots),
                discovery_interval=self.discovery_interval,
                snapshot_interval=self.snapshot_interval,
            )

            tasks = [
                self._discovery_loop(),
                self._snapshot_loop(),
                self._resolution_loop(),
                self._db_writer_loop(),
            ]

            try:
                await asyncio.gather(*tasks)
            finally:
                self._http_client = None
                await self.chainlink.disconnect()
                await self.polymarket.disconnect()
                await self._engine.dispose()


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Passive slot data collector")
    p.add_argument("--symbols", type=str, default="BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT")
    p.add_argument("--slot-duration", type=str, default="15m",
                    choices=list(DURATION_MAP.keys()),
                    help="Slot duration: 5m or 15m (default: 15m)")
    p.add_argument("--db-url", type=str,
                    default=settings.DATABASE_URL,
                    help="Database URL (mysql+aiomysql:// or sqlite+aiosqlite://)")
    p.add_argument("--discovery-interval", type=float, default=60.0)
    p.add_argument("--snapshot-interval", type=float, default=30.0)
    p.add_argument("--resolution-interval", type=float, default=60.0)
    p.add_argument("--strategy-tag", type=str, default="slot_collector")
    return p


async def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    slot_duration = DURATION_MAP[args.slot_duration]

    symbols = [
        s if s.endswith("USDT") else s + "USDT"
        for s in (tok.strip().upper() for tok in args.symbols.split(","))
        if s
    ]

    db_url = args.db_url
    if not db_url:
        print("ERROR: --db-url required (or set DATABASE_URL in .env)")
        return

    polymarket = PolymarketFeed()
    chainlink = ChainlinkFeed()

    collector = SlotCollector(
        polymarket=polymarket,
        chainlink=chainlink,
        symbols=symbols,
        db_url=db_url,
        slot_duration=slot_duration,
        discovery_interval=args.discovery_interval,
        snapshot_interval=args.snapshot_interval,
        resolution_interval=args.resolution_interval,
    )

    sym_names = [s.replace("USDT", "") for s in symbols]
    print(f"=== Slot Collector ({args.slot_duration}) ===")
    print(f"  Symbols:     {', '.join(sym_names)}")
    print(f"  Duration:    {args.slot_duration} ({slot_duration}s)")
    print(f"  Database:    {db_url.split('@')[-1] if '@' in db_url else db_url[:40]}")
    print(f"  Snapshot:    every {args.snapshot_interval}s")
    print(f"  Discovery:   every {args.discovery_interval}s")
    print(f"  Resolution:  every {args.resolution_interval}s")
    print()

    await collector.run()


if __name__ == "__main__":
    if uvloop is not None:
        uvloop.install()
    asyncio.run(main())
