#!/usr/bin/env python3
"""ML/XGBoost-driven TD maker for Polymarket crypto markets.

Same infrastructure as run_crypto_td_maker.py but with ML model controlling
entry AND exit decisions. Reuses all src/td_maker/ components — only the
EntryFilters and StopLossManager are wired with an ML model instead of
rule-based thresholds.

ML model responsibilities:
  - Entry: model.predict(market, outcome, bid) → P(win) ∈ [0, 1]
    - Below hybrid_skip_below (0.55): skip
    - Above hybrid_taker_above (0.72): take liquidity
    - Otherwise: place maker order; Sizing uses p_win for Kelly scaling
  - Exit: exit_model.predict_exit(market, bid) → P(should_exit) ∈ [0, 1]
    - Above exit_threshold (0.35): trigger sell

Models are loaded via joblib. See src/td_maker/filters.py for the interface.

USAGE:
    ./run scripts/run_crypto_td_lm.py --paper --model-path models/entry.pkl
    ./run scripts/run_crypto_td_lm.py --live --model-path models/entry.pkl \\
        --exit-model-path models/exit.pkl --exit-threshold 0.4
"""
from __future__ import annotations

import argparse
import asyncio

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

from scripts.run_crypto_td_maker import (
    DBFire, MagicExecutor, AsyncUserFeed, _auto_detect_wallet,
    _load_model, _load_exit_model,
)

logger = structlog.get_logger()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="ML/XGBoost-driven TD maker — entry and exit via model")

    # Mode
    p.add_argument("--paper", action="store_true", default=True)
    p.add_argument("--live", action="store_true", default=False)
    p.add_argument("--autopilot", action="store_true", default=False)

    # Market
    p.add_argument("--symbols", type=str,
                   default="BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT")
    p.add_argument("--slot-minutes", type=int, default=15, dest="slot_minutes",
                   help="Slot duration in minutes (5 or 15)")

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

    # Filters (basic, ML overrides entry decision)
    p.add_argument("--min-entry-minutes", type=float, default=0.0,
                   dest="min_entry_minutes")
    p.add_argument("--max-entry-minutes", type=float, default=0.0,
                   dest="max_entry_minutes")
    p.add_argument("--min-move-pct", type=float, default=0.0,
                   dest="min_move_pct")
    p.add_argument("--max-move-pct", type=float, default=0.0,
                   dest="max_move_pct")
    p.add_argument("--entry-fair-margin", type=float, default=0.0,
                   dest="entry_fair_margin")
    p.add_argument("--min-book-depth", type=float, default=0.0,
                   dest="min_book_depth")
    p.add_argument("--avoid-hours-utc", type=int, nargs="*", default=[],
                   dest="avoid_hours_utc")

    # ML entry model — REQUIRED for this script to be useful
    p.add_argument("--model-path", default="", dest="model_path",
                   help="Path to joblib-serialized entry model (XGBoost etc)")
    p.add_argument("--hybrid-skip-below", type=float, default=0.55,
                   dest="hybrid_skip_below",
                   help="P(win) below this → skip entry")
    p.add_argument("--hybrid-taker-above", type=float, default=0.72,
                   dest="hybrid_taker_above",
                   help="P(win) above this → take liquidity (taker)")

    # Stop-loss (rule-based fallback even with ML exit)
    p.add_argument("--stoploss-peak", type=float, default=0.0,
                   dest="stoploss_peak")
    p.add_argument("--stoploss-exit", type=float, default=0.0,
                   dest="stoploss_exit")
    p.add_argument("--stoploss-fair-margin", type=float, default=0.10,
                   dest="stoploss_fair_margin")

    # ML exit model
    p.add_argument("--exit-model-path", default="", dest="exit_model_path",
                   help="Path to joblib-serialized exit model")
    p.add_argument("--exit-threshold", type=float, default=0.35,
                   dest="exit_threshold",
                   help="P(exit) threshold to trigger sell")

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
    p.add_argument("--strategy-tag", type=str, default="crypto_td_lm",
                   dest="strategy_tag")
    p.add_argument("--db-url", type=str, default=settings.DATABASE_URL,
                   dest="db_url")

    return p


class Config:
    def __init__(self, args: argparse.Namespace) -> None:
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.paper_mode = not (args.live or args.autopilot)
        self.symbols = [s.strip() for s in args.symbols.split(",")]
        self.slot_duration = args.slot_minutes * 60
        if not self.db_url:
            self.db_url = settings.DATABASE_URL or ""


async def main_async() -> None:
    args = build_parser().parse_args()
    config = Config(args)

    if not config.model_path:
        logger.warning("no_model_path_specified",
                       msg="Running without ML model — filters will be rule-based only")

    executor = PolymarketExecutor(settings) if not config.paper_mode else MagicExecutor()
    poly_feed = PolymarketFeed(settings)
    user_feed = PolymarketUserFeed(settings) if not config.paper_mode else AsyncUserFeed()
    chainlink_feed = ChainlinkFeed(settings)
    shadow = TakerShadow()

    if config.wallet <= 0 and not config.paper_mode:
        config.wallet = await _auto_detect_wallet(executor)
        logger.info("wallet_auto_detected", balance=config.wallet)
    if config.order_size <= 0:
        config.order_size = max(1.0, config.wallet * 0.025)
    if config.max_exposure <= 0:
        config.max_exposure = max(config.order_size, config.wallet * 0.50)

    # Load ML models
    entry_model = _load_model(config.model_path)
    exit_model = _load_exit_model(config.exit_model_path)

    if entry_model is None:
        logger.info("entry_model_not_loaded",
                    msg="Falling back to rule-based entry filters")
    else:
        logger.info("entry_model_loaded", path=config.model_path)

    if exit_model is not None:
        logger.info("exit_model_loaded", path=config.exit_model_path)

    guard = RiskGuard(
        max_consecutive_losses=config.cb_max_losses,
        max_drawdown=config.cb_max_drawdown,
        daily_loss_limit=config.cb_daily_limit,
        stale_threshold=config.cb_stale_seconds,
        poly_feed=poly_feed,
        db_url=config.db_url,
    )
    manager = TradeManager(
        executor=executor,
        guard=guard,
        settings=settings,
        strategy_tag=config.strategy_tag,
        paper_mode=config.paper_mode,
    )
    db = DBFire(config.db_url, config.strategy_tag)

    registry = MarketRegistry()
    # EntryFilters uses model if provided: predict → P(win) → maker/taker/skip
    filters = EntryFilters(chainlink_feed, config, model=entry_model)
    # Sizing uses p_win from FilterResult for Kelly-scaled position sizing
    sizing = Sizing(config, model=entry_model)
    order_mgr = OrderManager(executor=executor,
                             registry=registry, db=db,
                             trade_manager=manager, config=config)
    fill_detector = FillDetector(
        registry=registry, order_mgr=order_mgr,
        poly_feed=poly_feed, user_feed=user_feed,
        trade_manager=manager, shadow=shadow,
        db=db, config=config, executor=executor)
    # StopLossManager: exit_model overrides rule-based stop-loss if provided
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

    model_desc = f"entry={config.model_path or 'rules'} exit={config.exit_model_path or 'rules'}"
    print(f"TD Maker LM starting — paper={config.paper_mode} "
          f"size=${config.order_size:.0f} max_exp=${config.max_exposure:.0f} "
          f"slot={config.slot_minutes}min | {model_desc}")

    await engine.run()


def main() -> None:
    if uvloop:
        uvloop.install()
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
