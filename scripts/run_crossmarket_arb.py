#!/usr/bin/env python3
# scripts/run_crossmarket_arb.py
"""
Cross-Market Arbitrage Bot - Main Entry Point

Usage:
    python scripts/run_crossmarket_arb.py [--autopilot] [--scan-interval SECONDS]

This script:
1. Connects to Azuro, Overtime, and Polymarket feeds
2. Matches identical events across platforms using LLM verification
3. Scans for price discrepancies (arbitrage opportunities)
4. Sends Telegram alerts (or auto-executes in autopilot mode)
"""

import asyncio
import argparse
import signal
import sys
from pathlib import Path
from typing import Any, Optional

# Add project root to path for imports
_project_root = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root))

import structlog

# Setup logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.dev.ConsoleRenderer()
    ]
)
logger = structlog.get_logger()

from config.settings import settings
from src.feeds.azuro import AzuroFeed
from src.feeds.overtime import OvertimeFeed
from src.feeds.polymarket import PolymarketFeed
from src.matching.event_matcher import CrossMarketMatcher, MatchedEvent
from src.arb.cross_market_arb import CrossMarketArbEngine, EvaluationResult
from src.risk.manager import UnifiedRiskManager
from src.bot.crossmarket_handlers import CrossMarketArbHandler
from src.db.database import init_db_async


class CrossMarketArbBot:
    """Main bot orchestrator for cross-market arbitrage.

    Coordinates feeds, event matching, opportunity detection, and execution.
    """

    def __init__(
        self,
        autopilot: bool = False,
        scan_interval: float = 5.0,
    ):
        """Initialize the Cross-Market Arbitrage Bot.

        Args:
            autopilot: If True, auto-execute trades; if False, send Telegram alerts
            scan_interval: Seconds between scan cycles (default: 5.0)
        """
        self.autopilot = autopilot
        self.scan_interval = scan_interval

        # Initialize feeds
        self.azuro_feed = AzuroFeed()
        self.overtime_feed = OvertimeFeed()
        self.polymarket_feed = PolymarketFeed()

        # Initialize matcher
        self.matcher = CrossMarketMatcher()

        # Initialize risk manager
        self.risk_manager = UnifiedRiskManager(
            global_capital=settings.GLOBAL_CAPITAL,
            reality_allocation_pct=settings.CAPITAL_ALLOCATION_REALITY_PCT,
            crossmarket_allocation_pct=settings.CAPITAL_ALLOCATION_CROSSMARKET_PCT,
            max_position_pct=settings.MAX_POSITION_PCT,
            daily_loss_limit_pct=settings.DAILY_LOSS_LIMIT_PCT,
        )

        # Initialize arb engine
        self.engine = CrossMarketArbEngine(
            risk_manager=self.risk_manager,
            min_edge_pct=settings.CROSSMARKET_MIN_EDGE_PCT,
        )

        # Telegram handler (initialized later if configured)
        self.telegram_handler: Optional[CrossMarketArbHandler] = None
        self._telegram_bot: Optional[Any] = None

        # State
        self._running = False
        self._matched_events: list[MatchedEvent] = []

    async def start(self) -> None:
        """Start the bot and run the main scan loop."""
        logger.info(
            "starting_crossmarket_arb_bot",
            autopilot=self.autopilot,
            scan_interval=self.scan_interval,
        )

        # Initialize database
        await init_db_async()

        # Connect to feeds
        await self._connect_feeds()

        # Initialize Telegram if configured
        await self._init_telegram()

        self._running = True
        logger.info("bot_started")

        # Main scan loop
        await self._scan_loop()

    async def stop(self) -> None:
        """Stop the bot gracefully."""
        logger.info("stopping_bot")
        self._running = False

        # Disconnect feeds
        await self._disconnect_feeds()

        logger.info("bot_stopped")

    async def _connect_feeds(self) -> None:
        """Connect to all data feeds."""
        logger.info("connecting_feeds")

        try:
            await self.azuro_feed.connect()
            logger.info("azuro_feed_connected")
        except Exception as e:
            logger.error("azuro_feed_connection_failed", error=str(e))

        try:
            await self.overtime_feed.connect()
            logger.info("overtime_feed_connected")
        except Exception as e:
            logger.error("overtime_feed_connection_failed", error=str(e))

        try:
            await self.polymarket_feed.connect()
            logger.info("polymarket_feed_connected")
        except Exception as e:
            logger.error("polymarket_feed_connection_failed", error=str(e))

    async def _disconnect_feeds(self) -> None:
        """Disconnect from all data feeds."""
        logger.info("disconnecting_feeds")

        try:
            await self.azuro_feed.disconnect()
        except Exception as e:
            logger.warning("azuro_feed_disconnect_error", error=str(e))

        try:
            await self.overtime_feed.disconnect()
        except Exception as e:
            logger.warning("overtime_feed_disconnect_error", error=str(e))

        try:
            await self.polymarket_feed.disconnect()
        except Exception as e:
            logger.warning("polymarket_feed_disconnect_error", error=str(e))

    async def _init_telegram(self) -> None:
        """Initialize Telegram bot if configured."""
        if not settings.TELEGRAM_BOT_TOKEN or not settings.TELEGRAM_CHAT_ID:
            logger.warning("telegram_not_configured")
            return

        try:
            from telegram import Bot

            self._telegram_bot = Bot(token=settings.TELEGRAM_BOT_TOKEN)
            self.telegram_handler = CrossMarketArbHandler(
                bot=self._telegram_bot,
                chat_id=settings.TELEGRAM_CHAT_ID,
            )
            logger.info("telegram_initialized")
        except ImportError:
            logger.warning("telegram_package_not_installed")
        except Exception as e:
            logger.error("telegram_initialization_failed", error=str(e))

    async def _scan_loop(self) -> None:
        """Main scanning loop that finds and processes opportunities."""
        while self._running:
            try:
                await self._run_scan_cycle()
            except Exception as e:
                logger.error("scan_cycle_error", error=str(e))

            await asyncio.sleep(self.scan_interval)

    async def _run_scan_cycle(self) -> None:
        """Run a single scan cycle."""
        logger.debug("starting_scan_cycle")

        # Step 1: Fetch events from all platforms
        azuro_events = await self._fetch_azuro_events()
        overtime_games = await self._fetch_overtime_games()
        polymarket_markets = await self._fetch_polymarket_markets()

        # Step 2: Match events across platforms
        matched_events = await self.matcher.match_all(
            polymarket_events=polymarket_markets,
            azuro_events=azuro_events,
            overtime_games=overtime_games,
        )

        if not matched_events:
            logger.debug("no_matched_events_found")
            return

        # Store matched events
        self._matched_events = matched_events
        logger.info("events_matched", count=len(matched_events))

        # Step 3: Get prices for matched events
        price_feeds = await self._get_prices_for_events(matched_events)

        # Step 4: Scan for arbitrage opportunities
        results = await self.engine.scan_all_events(
            matched_events=matched_events,
            price_feeds=price_feeds,
        )

        # Step 5: Handle each valid opportunity
        for result in results:
            if result.approved:
                await self._handle_opportunity(result)

    async def _fetch_azuro_events(self) -> list:
        """Fetch active events from Azuro.

        Returns:
            List of AzuroEvent objects
        """
        try:
            events = await self.azuro_feed.get_active_events(page_size=100)
            logger.debug("azuro_events_fetched", count=len(events))
            return events
        except Exception as e:
            logger.error("azuro_fetch_failed", error=str(e))
            return []

    async def _fetch_overtime_games(self) -> list:
        """Fetch active games from Overtime.

        Returns:
            List of OvertimeGame objects
        """
        try:
            games = await self.overtime_feed.get_active_games(page_size=100)
            logger.debug("overtime_games_fetched", count=len(games))
            return games
        except Exception as e:
            logger.error("overtime_fetch_failed", error=str(e))
            return []

    async def _fetch_polymarket_markets(self) -> list[dict[str, Any]]:
        """Fetch active markets from Polymarket.

        Note: Polymarket is primarily used for price discovery on matched events.
        The matching primarily happens between Azuro and Overtime for sports events.

        Returns:
            List of Polymarket market dictionaries
        """
        # Polymarket markets are typically fetched via their REST API
        # For now, return empty list - Polymarket integration is optional
        # The bot can work with just Azuro and Overtime for sports betting
        logger.debug("polymarket_markets_fetch_skipped")
        return []

    async def _get_prices_for_events(
        self,
        matched_events: list[MatchedEvent],
    ) -> dict[str, dict[str, dict[str, float]]]:
        """Get current prices for all matched events.

        Args:
            matched_events: List of matched events across platforms

        Returns:
            Nested dict: event_id -> platform -> outcome -> price
        """
        price_feeds: dict[str, dict[str, dict[str, float]]] = {}

        for event in matched_events:
            event_id = self._get_event_id(event)
            prices: dict[str, dict[str, float]] = {}

            # Get Azuro prices
            if event.azuro_event:
                outcomes = event.azuro_event.outcomes
                # Azuro outcomes are keyed by outcome_id, map to YES/NO
                # Typically: first outcome is YES (home win), second is NO (away win)
                outcome_ids = list(outcomes.keys())
                if len(outcome_ids) >= 2:
                    prices["azuro"] = {
                        "YES": outcomes[outcome_ids[0]],
                        "NO": outcomes[outcome_ids[1]],
                    }

            # Get Overtime prices
            if event.overtime_game:
                game = event.overtime_game
                prices["overtime"] = {
                    "YES": game.home_odds,
                    "NO": game.away_odds,
                }

            # Get Polymarket prices (if connected and subscribed)
            if event.polymarket_id:
                pm_prices = self.polymarket_feed.get_market_prices(event.polymarket_id)
                if pm_prices:
                    yes_bid, yes_ask = pm_prices.get("YES", (None, None))
                    no_bid, no_ask = pm_prices.get("NO", (None, None))

                    pm_entry: dict[str, float] = {}
                    if yes_bid is not None and yes_ask is not None:
                        pm_entry["YES"] = (yes_bid + yes_ask) / 2
                    if no_bid is not None and no_ask is not None:
                        pm_entry["NO"] = (no_bid + no_ask) / 2

                    if pm_entry:
                        prices["polymarket"] = pm_entry

            if prices:
                price_feeds[event_id] = prices

        logger.debug("prices_collected", events_with_prices=len(price_feeds))
        return price_feeds

    def _get_event_id(self, event: MatchedEvent) -> str:
        """Get a unique identifier for a matched event.

        Args:
            event: The matched event

        Returns:
            String identifier
        """
        if event.polymarket_id:
            return event.polymarket_id
        if event.azuro_condition_id:
            return event.azuro_condition_id
        if event.overtime_game_id:
            return event.overtime_game_id
        return event.name

    async def _handle_opportunity(self, result: EvaluationResult) -> None:
        """Handle a detected arbitrage opportunity.

        Args:
            result: The evaluation result with opportunity details
        """
        opportunity = result.opportunity
        position_size = result.position_size

        logger.info(
            "opportunity_detected",
            event_name=getattr(opportunity.event, "name", str(opportunity.event)),
            source_platform=opportunity.source_platform,
            target_platform=opportunity.target_platform,
            net_edge_pct=opportunity.net_edge_pct,
            position_size=position_size,
        )

        if self.autopilot:
            # Auto-execute the trade
            await self._execute_trade(opportunity, position_size)
        else:
            # Send Telegram alert
            await self._send_alert(opportunity, position_size)

    async def _execute_trade(
        self,
        opportunity: Any,
        position_size: float,
    ) -> None:
        """Execute a cross-market arbitrage trade.

        Args:
            opportunity: The arbitrage opportunity
            position_size: Position size in USD
        """
        logger.info(
            "executing_trade",
            source_platform=opportunity.source_platform,
            target_platform=opportunity.target_platform,
            position_size=position_size,
        )

        # TODO: Implement actual trade execution
        # This would involve:
        # 1. Place buy order on source platform
        # 2. Place sell order on target platform
        # 3. Confirm both executions
        # 4. Record P&L

        logger.info("trade_execution_placeholder")

    async def _send_alert(
        self,
        opportunity: Any,
        position_size: float,
    ) -> None:
        """Send a Telegram alert for the opportunity.

        Args:
            opportunity: The arbitrage opportunity
            position_size: Suggested position size in USD
        """
        if not self.telegram_handler:
            logger.warning("telegram_not_available_for_alert")
            return

        try:
            message_id = await self.telegram_handler.send_alert(
                opportunity=opportunity,
                position_size=position_size,
            )

            if message_id:
                logger.info("alert_sent", message_id=message_id)
            else:
                logger.warning("alert_send_failed")

        except Exception as e:
            logger.error("alert_send_error", error=str(e))


async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Cross-Market Arbitrage Bot")
    parser.add_argument(
        "--autopilot",
        action="store_true",
        help="Enable autopilot mode (auto-execute trades)",
    )
    parser.add_argument(
        "--scan-interval",
        type=float,
        default=settings.CROSSMARKET_SCAN_INTERVAL_SECONDS,
        help=f"Seconds between scans (default: {settings.CROSSMARKET_SCAN_INTERVAL_SECONDS})",
    )

    args = parser.parse_args()

    bot = CrossMarketArbBot(
        autopilot=args.autopilot,
        scan_interval=args.scan_interval,
    )

    # Handle shutdown signals
    loop = asyncio.get_running_loop()

    def signal_handler() -> None:
        logger.info("shutdown_signal_received")
        asyncio.create_task(bot.stop())

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    try:
        await bot.start()
    except KeyboardInterrupt:
        await bot.stop()
    except Exception as e:
        logger.error("bot_error", error=str(e))
        await bot.stop()
        raise


if __name__ == "__main__":
    asyncio.run(main())
