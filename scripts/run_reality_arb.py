#!/usr/bin/env python3
# scripts/run_reality_arb.py
"""
Reality Arbitrage Bot - Main Entry Point

Usage:
    python scripts/run_reality_arb.py [--game lol|dota2|csgo] [--autopilot]

This script:
1. Connects to PandaScore for live esports events
2. Connects to Polymarket WebSocket for order book
3. Monitors for significant game events
4. Detects arbitrage opportunities from broadcast lag
5. Sends Telegram alerts (or auto-executes in autopilot mode)
"""

import asyncio
import argparse
import signal
from typing import Any, Optional

import structlog

from src.utils.logging import configure_logging

configure_logging()
logger = structlog.get_logger()

from config.settings import settings
from src.feeds.base import FeedEvent
from src.feeds.pandascore import PandaScoreFeed
from src.feeds.polymarket import PolymarketFeed
from src.realtime.event_detector import EventDetector
from src.realtime.market_mapper import MarketMapper, MarketMapping
from src.arb.reality_arb import RealityArbEngine
from src.arb.polymarket_executor import PolymarketExecutor
from src.db.database import init_db_async
from src.bot.reality_handlers import RealityArbHandler
from src.risk.guard import RiskGuard


class RealityArbBot:
    """Main bot orchestrator."""

    def __init__(self, game: str = "lol", autopilot: bool = False):
        """Initialize the Reality Arbitrage Bot.

        Args:
            game: Game to monitor (lol, dota2, csgo)
            autopilot: If True, auto-execute trades; if False, send Telegram alerts
        """
        self.game = game
        self.autopilot = autopilot

        # Allocated capital for this strategy
        self.allocated_capital = (
            settings.GLOBAL_CAPITAL * (settings.CAPITAL_ALLOCATION_REALITY_PCT / 100.0)
        )

        # Components
        self.pandascore = PandaScoreFeed(api_key=settings.PANDASCORE_API_KEY)
        self.polymarket = PolymarketFeed()
        self.detector = EventDetector()
        self.mapper = MarketMapper()
        self.guard: RiskGuard | None = None
        self.engine = RealityArbEngine(
            polymarket_feed=self.polymarket,
            event_detector=self.detector,
            market_mapper=self.mapper,
            allocated_capital=self.allocated_capital,
            autopilot=self.autopilot,
        )

        # Telegram handler (optional)
        self.telegram_handler: Optional[RealityArbHandler] = None
        self._telegram_bot: Optional[Any] = None

        # Latest frame snapshots by match_id
        self._latest_frames: dict[str, dict[str, Any]] = {}

        self._running = False

    async def start(self) -> None:
        """Start the bot."""
        logger.info("starting_reality_arb_bot", game=self.game, autopilot=self.autopilot)

        # Initialize database
        await init_db_async()

        # Initialize RiskGuard
        self.guard = RiskGuard(
            strategy_tag="reality_arb",
            db_path="data/arb.db",
            daily_loss_limit_usd=-(
                settings.GLOBAL_CAPITAL * settings.DAILY_LOSS_LIMIT_PCT
            ),
        )
        await self.guard.initialize()
        self.engine.guard = self.guard

        # Connect to feeds
        await self.pandascore.connect()
        await self.polymarket.connect()

        # Sync markets from Polymarket for mapping
        await self._sync_markets()

        # Initialize executor and Telegram if configured
        await self._init_executor()
        await self._init_telegram()

        # Register event handler
        self.pandascore.on_event(self._on_game_event)

        # Discover and subscribe to live matches
        await self._subscribe_live_matches()

        self._running = True
        logger.info("bot_started")

        # Keep running
        while self._running:
            # Periodically refresh live matches
            await asyncio.sleep(60)
            await self._subscribe_live_matches()

    async def stop(self) -> None:
        """Stop the bot gracefully."""
        logger.info("stopping_bot")
        self._running = False

        await self.pandascore.disconnect()
        await self.polymarket.disconnect()

        logger.info("bot_stopped")

    async def _sync_markets(self) -> None:
        """Sync Polymarket markets into the mapper."""
        try:
            new_count = self.mapper.sync_from_polymarket(
                game=self.game,
                limit=settings.REALITY_SYNC_MARKET_LIMIT,
                only_active=settings.REALITY_SYNC_ONLY_ACTIVE,
            )
            logger.info(
                "polymarket_sync_complete",
                game=self.game,
                new_mappings=new_count,
            )
        except Exception as e:
            logger.error("polymarket_sync_failed", error=str(e))

    async def _init_executor(self) -> None:
        """Initialize Polymarket executor if credentials are configured."""
        try:
            self.engine.executor = PolymarketExecutor.from_settings()
            logger.info("polymarket_executor_initialized")
        except Exception as e:
            logger.warning("polymarket_executor_not_configured", error=str(e))

    async def _init_telegram(self) -> None:
        """Initialize Telegram bot if configured."""
        if not settings.TELEGRAM_BOT_TOKEN or not settings.TELEGRAM_CHAT_ID:
            logger.warning("telegram_not_configured")
            return

        try:
            from telegram import Bot

            self._telegram_bot = Bot(token=settings.TELEGRAM_BOT_TOKEN)
            self.telegram_handler = RealityArbHandler(
                bot=self._telegram_bot,
                engine=self.engine,
            )
            logger.info("telegram_initialized")
        except ImportError:
            logger.warning("telegram_package_not_installed")
        except Exception as e:
            logger.error("telegram_initialization_failed", error=str(e))

    async def _subscribe_live_matches(self) -> None:
        """Find and subscribe to currently live matches."""
        try:
            matches = await self.pandascore.get_live_matches(self.game)
        except Exception as e:
            logger.error("get_live_matches_failed", error=str(e))
            return

        for match in matches:
            match_id = str(match.get("id", ""))
            if not match_id:
                continue

            # Check if we have a Polymarket mapping for this match
            teams = self._extract_teams(match)
            league = match.get("league", {}).get("name")

            mapping = self.mapper.find_market(
                game=self.game,
                teams=teams,
                league=league
            )
            if not mapping and len(teams) >= 2:
                mapping = self.mapper.find_market_fuzzy(
                    game=self.game,
                    team_a=teams[0],
                    team_b=teams[1],
                )

            if mapping:
                self.mapper.link_match(match_id, mapping)
                await self.pandascore.subscribe(self.game, match_id)
                if settings.REALITY_USE_FRAMES:
                    await self.pandascore.subscribe_frames(self.game, match_id)

                token_ids = set(mapping.outcome_token_ids.values())
                if not token_ids:
                    token_ids = {mapping.polymarket_id}

                for token_id in token_ids:
                    await self.polymarket.subscribe_market(token_id)

                logger.info("subscribed_to_match",
                           match=match.get("name"),
                           market=mapping.polymarket_id)

    def _extract_teams(self, match: dict[str, Any]) -> list[str]:
        """Extract team names from match data.

        Args:
            match: Match data from PandaScore API

        Returns:
            List of team names
        """
        opponents = match.get("opponents", [])
        return [opp.get("opponent", {}).get("name", "Unknown") for opp in opponents]

    async def _on_game_event(self, event: Any) -> None:
        """Handle incoming game event.

        Args:
            event: Game event from PandaScore feed
        """
        # Frame events are used to enrich later events
        if getattr(event, "event_type", "") == "frame":
            if getattr(event, "match_id", None):
                self._latest_frames[str(event.match_id)] = event.data
            return

        event_to_classify = event
        match_id = getattr(event, "match_id", None)
        if match_id and str(match_id) in self._latest_frames:
            merged_data = {
                **self._latest_frames[str(match_id)],
                **event.data,
            }
            event_to_classify = FeedEvent(
                source=event.source,
                event_type=event.event_type,
                game=event.game,
                data=merged_data,
                timestamp=event.timestamp,
                match_id=str(match_id),
            )

        # Classify event
        significant = self.detector.classify(event_to_classify)

        if not significant.should_trade:
            return

        logger.info("significant_event_detected",
                   event=significant.event_description,
                   impact=significant.impact_score)

        # Find market mapping - use match_id from the event if available
        mapping: Optional[MarketMapping] = None

        if match_id:
            mapping = self.mapper.find_by_match_id(str(match_id))

        if not mapping:
            logger.debug("no_market_mapping_found", match_id=match_id)
            return

        # Evaluate opportunity
        result = await self.engine.process_event(significant, mapping)

        if not result:
            return

        status = result.get("status")
        if status in ("RISK_HALTED", "RATE_LIMITED", "SIZE_ZERO"):
            logger.info("opportunity_skipped", status=status)
            return

        if not self.autopilot:
            # Send Telegram alert for manual approval
            if self.telegram_handler and settings.TELEGRAM_CHAT_ID:
                if status in ("PENDING_APPROVAL", "CLOSE_PENDING_APPROVAL"):
                    pending_opp = self.engine._pending_opportunities.get(
                        mapping.polymarket_id
                    )
                    if pending_opp:
                        try:
                            chat_id = int(settings.TELEGRAM_CHAT_ID)
                            size = result.get("size", 0.0)
                            await self.telegram_handler.send_alert(
                                chat_id=chat_id,
                                opportunity=pending_opp,
                                market_title=mapping.event_identifier,
                                size=size,
                            )
                        except ValueError:
                            logger.error(
                                "invalid_telegram_chat_id",
                                chat_id=settings.TELEGRAM_CHAT_ID,
                            )


async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Reality Arbitrage Bot")
    parser.add_argument("--game", choices=["lol", "dota2", "csgo"], default="lol",
                       help="Game to monitor (default: lol)")
    parser.add_argument("--autopilot", action="store_true",
                       help="Enable autopilot mode (auto-execute trades)")

    args = parser.parse_args()

    bot = RealityArbBot(game=args.game, autopilot=args.autopilot)

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
