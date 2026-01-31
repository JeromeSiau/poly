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
import os
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
from src.feeds.pandascore import PandaScoreFeed
from src.feeds.polymarket import PolymarketFeed
from src.realtime.event_detector import EventDetector, SignificantEvent
from src.realtime.market_mapper import MarketMapper, MarketMapping
from src.arb.reality_arb import RealityArbEngine
from src.db.database import init_db_async
from src.bot.reality_handlers import RealityArbHandler


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

        # Components
        self.pandascore = PandaScoreFeed(api_key=settings.PANDASCORE_API_KEY)
        self.polymarket = PolymarketFeed()
        self.detector = EventDetector()
        self.mapper = MarketMapper()
        self.engine = RealityArbEngine(
            polymarket_feed=self.polymarket,
            event_detector=self.detector,
            market_mapper=self.mapper
        )

        # Telegram handler (optional)
        self.telegram_handler: Optional[RealityArbHandler] = None

        self._running = False

    async def start(self) -> None:
        """Start the bot."""
        logger.info("starting_reality_arb_bot", game=self.game, autopilot=self.autopilot)

        # Initialize database
        await init_db_async()

        # Connect to feeds
        await self.pandascore.connect()
        await self.polymarket.connect()

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

            if mapping:
                await self.pandascore.subscribe(self.game, match_id)
                await self.polymarket.subscribe_market(mapping.polymarket_id)

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
        # Classify event
        significant = self.detector.classify(event)

        if not significant.should_trade:
            return

        logger.info("significant_event_detected",
                   event=significant.event_description,
                   impact=significant.impact_score)

        # Find market mapping - use match_id from the event if available
        match_id = getattr(event, 'match_id', None)
        mapping: Optional[MarketMapping] = None

        if match_id:
            # Try to find mapping by looking through all mappings
            for m in self.mapper.get_all_mappings():
                if match_id in m.event_identifier:
                    mapping = m
                    break

        if not mapping:
            logger.debug("no_market_mapping_found", match_id=match_id)
            return

        # Evaluate opportunity
        opportunity = await self.engine.process_event(significant, mapping)

        if opportunity:
            size = self.engine.calculate_position_size(
                self.engine._pending_opportunities.get(mapping.polymarket_id)
            ) if mapping.polymarket_id in self.engine._pending_opportunities else 0.0

            if self.autopilot:
                # Auto-execute
                pending_opp = self.engine._pending_opportunities.get(mapping.polymarket_id)
                if pending_opp:
                    result = await self.engine.execute(pending_opp, size)
                    logger.info("auto_executed", result=result)
            else:
                # Send Telegram alert
                if self.telegram_handler and settings.TELEGRAM_CHAT_ID:
                    pending_opp = self.engine._pending_opportunities.get(mapping.polymarket_id)
                    if pending_opp:
                        try:
                            chat_id = int(settings.TELEGRAM_CHAT_ID)
                            await self.telegram_handler.send_alert(
                                chat_id=chat_id,
                                opportunity=pending_opp,
                                market_title=mapping.event_identifier,
                                size=size
                            )
                        except ValueError:
                            logger.error("invalid_telegram_chat_id",
                                       chat_id=settings.TELEGRAM_CHAT_ID)


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
