#!/usr/bin/env python3
"""Passive time-decay maker for Polymarket 15-min crypto markets.

Entry point — wires dependencies, starts TDMakerEngine.
See src/td_maker/ for business logic.

STRATEGY (validated by backtest_crypto_minute.py):
    On Polymarket 15-min crypto binary markets (BTC/ETH up/down), the
    favourite side (priced 0.75-0.85) wins ~1% more often than implied.
    Since Jan 2026, taker fees (~1% at p=0.80) eat this edge — but
    MAKER orders pay 0 fees and earn rebates.

USAGE:
    ./run scripts/run_crypto_td_maker.py --paper             # default (paper mode)
    ./run scripts/run_crypto_td_maker.py --live               # real orders
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone

import structlog

from src.utils.logging import configure_logging

configure_logging()

try:
    import uvloop
except ImportError:
    uvloop = None

from config.settings import settings
from src.arb.polymarket_executor import PolymarketExecutor
from src.execution import TradeManager
from src.feeds.chainlink import ChainlinkFeed
from src.feeds.polymarket import PolymarketFeed, PolymarketUserFeed
from src.risk.guard import RiskGuard
from src.shadow.taker_shadow import TakerShadow

from src.td_maker.state import MarketRegistry
from src.td_maker.engine import TDMakerEngine
from src.td_maker.discovery import MarketDiscovery
from src.td_maker.filters import EntryFilters
from src.td_maker.sizing import Sizing
from src.td_maker.order_manager import OrderManager
from src.td_maker.fill_detector import FillDetector
from src.td_maker.stop_loss import StopLossManager
from src.td_maker.settlement import SettlementManager
from src.td_maker.bidding import BiddingEngine
from src.td_maker.status import StatusLine

logger = structlog.get_logger()

TD_MAKER_EVENT_TYPE = "crypto_td_maker"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="TD Maker strategy — 15-min crypto markets")

    # Mode
    p.add_argument("--paper", action="store_true", default=True,
                   help="Paper trading mode (default)")
    p.add_argument("--live", action="store_true", default=False,
                   help="Live trading mode")
    p.add_argument("--autopilot", action="store_true", default=False,
                   help="Alias for --live")

    # Market
    p.add_argument("--symbols", type=str,
                   default="BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT")

    # Sizing
    p.add_argument("--wallet", type=float, default=0.0)
    p.add_argument("--order-size", type=float, default=0.0, dest="order_size")
    p.add_argument("--max-exposure", type=float, default=0.0,
                   dest="max_exposure")

    # Ladder
    p.add_argument("--target-bid", type=float, default=0.75,
                   dest="target_bid")
    p.add_argument("--max-bid", type=float, default=0.85, dest="max_bid")
    p.add_argument("--ladder-rungs", type=int, default=1,
                   dest="ladder_rungs")

    # Timing
    p.add_argument("--discovery-interval", type=float, default=60.0,
                   dest="discovery_interval")
    p.add_argument("--maker-interval", type=float, default=0.5,
                   dest="maker_interval")

    # Filters
    p.add_argument("--min-move-pct", type=float, default=0.0,
                   dest="min_move_pct")
    p.add_argument("--max-move-pct", type=float, default=0.0,
                   dest="max_move_pct")
    p.add_argument("--min-entry-minutes", type=float, default=0.0,
                   dest="min_entry_minutes")
    p.add_argument("--max-entry-minutes", type=float, default=0.0,
                   dest="max_entry_minutes")
    p.add_argument("--entry-fair-margin", type=float, default=0.0,
                   dest="entry_fair_margin")
    p.add_argument("--min-book-depth", type=float, default=0.0,
                   dest="min_book_depth")
    p.add_argument("--avoid-hours-utc", type=int, nargs="*", default=[],
                   dest="avoid_hours_utc")

    # ML entry
    p.add_argument("--model-path", default="", dest="model_path")
    p.add_argument("--hybrid-skip-below", type=float, default=0.55,
                   dest="hybrid_skip_below")
    p.add_argument("--hybrid-taker-above", type=float, default=0.72,
                   dest="hybrid_taker_above")

    # Stop-loss
    p.add_argument("--stoploss-peak", type=float, default=0.0,
                   dest="stoploss_peak")
    p.add_argument("--stoploss-exit", type=float, default=0.0,
                   dest="stoploss_exit")
    p.add_argument("--stoploss-fair-margin", type=float, default=0.10,
                   dest="stoploss_fair_margin")

    # ML exit
    p.add_argument("--exit-model-path", default="", dest="exit_model_path")
    p.add_argument("--exit-threshold", type=float, default=0.35,
                   dest="exit_threshold")

    # Circuit breaker
    p.add_argument("--cb-max-losses", type=int, default=5,
                   dest="cb_max_losses")
    p.add_argument("--cb-max-drawdown", type=float, default=-50.0,
                   dest="cb_max_drawdown")
    p.add_argument("--cb-stale-seconds", type=float, default=30.0,
                   dest="cb_stale_seconds")
    p.add_argument("--cb-stale-cancel", type=float, default=120.0,
                   dest="cb_stale_cancel")
    p.add_argument("--cb-stale-exit", type=float, default=300.0,
                   dest="cb_stale_exit")
    p.add_argument("--cb-daily-limit", type=float, default=-200.0,
                   dest="cb_daily_limit")

    # Misc
    p.add_argument("--strategy-tag", type=str, default="crypto_td_maker",
                   dest="strategy_tag")
    p.add_argument("--db-url", type=str, default=settings.DATABASE_URL,
                   dest="db_url")

    return p


class Config:
    """Flat config object built from parsed args + derived values."""

    def __init__(self, args: argparse.Namespace) -> None:
        # Copy all args
        for k, v in vars(args).items():
            setattr(self, k, v)
        # Normalize paper_mode: live/autopilot override --paper
        self.paper_mode = not (args.live or args.autopilot)
        # Normalize symbols
        self.symbols = [s.strip() for s in args.symbols.split(",")]
        # slot_duration for 15-min markets
        self.slot_duration = 15 * 60
        # db_url fallback
        if not self.db_url:
            self.db_url = settings.DATABASE_URL or ""


async def _auto_detect_wallet(executor: PolymarketExecutor) -> float:
    try:
        balance = await executor.get_wallet_balance()
        return float(balance)
    except Exception:
        return 0.0


def _load_model(model_path: str):
    """Load XGBoost/ML entry model if path provided."""
    if not model_path:
        return None
    try:
        import joblib
        return joblib.load(model_path)
    except Exception as e:
        logger.warning("model_load_failed", path=model_path, error=str(e))
        return None


def _load_exit_model(model_path: str):
    """Load ML exit model if path provided."""
    if not model_path:
        return None
    try:
        import joblib
        return joblib.load(model_path)
    except Exception as e:
        logger.warning("exit_model_load_failed", path=model_path, error=str(e))
        return None


class DBFire:
    """Lazy fire-and-forget DB wrapper."""
    def __init__(self, db_url: str, strategy_tag: str):
        self.db_url = db_url
        self.strategy_tag = strategy_tag

    def fire(self, coro):
        asyncio.create_task(self._run(coro))

    async def _run(self, coro):
        try:
            await coro
        except Exception as e:
            logger.error("db_fire_error", error=str(e))

    def save_order(self, *, order_id, market, order):
        from src.db.td_orders import save_order
        return save_order(
            db_url=self.db_url, platform="polymarket",
            strategy_tag=self.strategy_tag,
            order_id=order_id, condition_id=market.condition_id,
            token_id=order.token_id, outcome=order.outcome,
            price=order.price, size_usd=order.size_usd,
            placed_at=order.placed_at)

    def mark_filled(self, order_id, *, shares):
        import time
        from src.db.td_orders import mark_filled
        return mark_filled(db_url=self.db_url,
                           order_id=order_id, shares=shares,
                           filled_at=time.time())

    def mark_settled(self, position, pnl):
        """Mark all order legs as settled, splitting PnL proportionally."""
        async def _settle_all():
            import time
            from src.db.td_orders import mark_settled
            now = time.time()
            legs = getattr(position, "order_legs", [])
            real_legs = [
                (oid, sz) for oid, sz in legs
                if not oid.startswith("_placing_") and oid
            ]
            if not real_legs:
                return
            total_size = sum(max(sz, 0.0) for _, sz in real_legs) or 1.0
            for oid, sz in real_legs:
                weight = max(sz, 0.0) / total_size
                try:
                    await mark_settled(db_url=self.db_url, order_id=oid,
                                       pnl=pnl * weight, settled_at=now)
                except Exception as e:
                    logger.error("db_settle_failed", oid=oid, error=str(e))
        return _settle_all()

    def delete_order(self, order_id):
        from src.db.td_orders import delete_order
        return delete_order(db_url=self.db_url, order_id=order_id)


class MagicExecutor:
    """Paper mode stub — generates fake order IDs, no real CLOB calls."""
    _counter: int = 0

    async def place_order(self, **kw):
        MagicExecutor._counter += 1
        return {"orderId": f"paper_{MagicExecutor._counter:06d}", "status": "LIVE"}

    async def cancel_order(self, *a): pass
    async def cancel_all_orders(self): return []
    async def get_order(self, *a): return None
    async def get_open_orders(self, **kw): return []
    async def get_wallet_balance(self): return 0.0


class AsyncUserFeed:
    """Paper mode stub for user feed."""

    class _Event:
        def is_set(self): return False
        def clear(self): pass
        def set(self): pass

    reconnected = _Event()
    is_connected = False

    async def connect(self): pass
    async def disconnect(self): pass
    async def subscribe_markets(self, *a): pass

    @property
    def fills(self):
        if not hasattr(self, "_q"):
            import asyncio
            self._q = asyncio.Queue()
        return self._q


async def main_async() -> None:
    args = build_parser().parse_args()
    config = Config(args)

    run_id = (f"{config.strategy_tag}-"
              f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")

    # Infrastructure
    executor = PolymarketExecutor(settings) if not config.paper_mode else MagicExecutor()
    poly_feed = PolymarketFeed(settings)
    chainlink_feed = ChainlinkFeed(settings)
    shadow = TakerShadow()

    # User feed (live only — needs per-credential constructor)
    if not config.paper_mode and settings.POLYMARKET_API_KEY:
        user_feed = PolymarketUserFeed(
            api_key=settings.POLYMARKET_API_KEY,
            api_secret=settings.POLYMARKET_API_SECRET,
            api_passphrase=settings.POLYMARKET_API_PASSPHRASE,
        )
    else:
        user_feed = AsyncUserFeed()

    # Trade manager
    manager = TradeManager(
        executor=executor if not config.paper_mode else None,
        strategy=config.strategy_tag,
        paper=config.paper_mode,
        db_url=config.db_url,
        event_type=TD_MAKER_EVENT_TYPE,
        run_id=run_id,
        notify_bids=False,
        notify_fills=not config.paper_mode,
        notify_closes=not config.paper_mode,
    )

    # Risk guard
    guard = RiskGuard(
        strategy_tag=config.strategy_tag,
        db_url=config.db_url,
        max_consecutive_losses=config.cb_max_losses,
        max_drawdown_usd=config.cb_max_drawdown,
        stale_seconds=config.cb_stale_seconds,
        stale_cancel_seconds=config.cb_stale_cancel,
        stale_exit_seconds=config.cb_stale_exit,
        daily_loss_limit_usd=config.cb_daily_limit,
        telegram_alerter=manager._alerter,
    )
    await guard.initialize()

    # Auto-sizing
    if config.wallet <= 0 and not config.paper_mode:
        config.wallet = await _auto_detect_wallet(executor)
        logger.info("wallet_auto_detected", balance=config.wallet)
    if config.order_size <= 0:
        config.order_size = max(1.0, config.wallet * 0.025)
    if config.max_exposure <= 0:
        config.max_exposure = max(config.order_size, config.wallet * 0.50)

    logger.info("sizing_configured",
                order_size=config.order_size,
                max_exposure=config.max_exposure)

    # Models
    entry_model = _load_model(config.model_path)
    exit_model = _load_exit_model(config.exit_model_path)

    db = DBFire(config.db_url, config.strategy_tag)

    # Components
    registry = MarketRegistry()
    filters = EntryFilters(chainlink_feed, config, model=entry_model)
    sizing = Sizing(config, model=entry_model)
    order_mgr = OrderManager(executor=executor,
                             registry=registry, db=db,
                             trade_manager=manager, config=config)
    fill_detector = FillDetector(
        registry=registry, order_mgr=order_mgr,
        poly_feed=poly_feed, user_feed=user_feed,
        trade_manager=manager, shadow=shadow,
        db=db, config=config, executor=executor)
    stop_loss = StopLossManager(
        registry=registry, order_mgr=order_mgr,
        executor=executor,
        chainlink_feed=chainlink_feed, trade_manager=manager,
        shadow=shadow, db=db, config=config,
        exit_model=exit_model, poly_feed=poly_feed)
    settlement = SettlementManager(
        registry=registry, executor=executor,
        trade_manager=manager, shadow=shadow, guard=guard,
        db=db, config=config, order_mgr=order_mgr)
    discovery = MarketDiscovery(poly_feed, chainlink_feed, config)
    bidding = BiddingEngine(registry=registry, filters=filters,
                            order_mgr=order_mgr, sizing=sizing,
                            config=config, poly_feed=poly_feed)
    status = StatusLine(registry, guard, shadow, config)

    engine = TDMakerEngine(
        registry=registry, discovery=discovery, bidding=bidding,
        order_mgr=order_mgr, fill_detector=fill_detector,
        stop_loss=stop_loss, settlement=settlement, status=status,
        guard=guard, poly_feed=poly_feed, user_feed=user_feed,
        chainlink_feed=chainlink_feed, config=config)

    print(f"TD Maker starting — paper={config.paper_mode} "
          f"size=${config.order_size:.0f} max_exp=${config.max_exposure:.0f} "
          f"rungs={config.ladder_rungs} [{config.target_bid}-{config.max_bid}]")

    await engine.run()


def main() -> None:
    if uvloop:
        uvloop.install()
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
