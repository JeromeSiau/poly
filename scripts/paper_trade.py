#!/usr/bin/env python3
# scripts/paper_trade.py
"""Run paper trading simulation with ML model predictions.

Usage:
    uv run python scripts/paper_trade.py \
        --model models/impact_model.pkl \
        --game lol \
        --capital 10000 \
        --telegram

This script:
1. Loads a trained ML model for win probability prediction
2. Connects to PandaScore feed for live esports events
3. Processes events through the PaperTradingEngine
4. Tracks paper trades and reports performance metrics
"""

import argparse
import asyncio
import signal
import sys
from datetime import datetime, timezone
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
        structlog.dev.ConsoleRenderer(),
    ]
)
logger = structlog.get_logger()

from config.settings import settings
from src.feeds.pandascore import PandaScoreFeed
from src.ml.train import ImpactModel
from src.paper_trading.engine import PaperTradingEngine


class PaperTradingBot:
    """Paper trading bot that uses ML model predictions."""

    def __init__(
        self,
        model: ImpactModel,
        game: str,
        capital: float,
        use_telegram: bool = False,
    ):
        """Initialize the paper trading bot.

        Args:
            model: Trained ImpactModel for predictions
            game: Game to monitor (lol, dota2, csgo, valorant)
            capital: Starting capital for paper trading
            use_telegram: Whether to send Telegram alerts
        """
        self.model = model
        self.game = game
        self.capital = capital
        self.use_telegram = use_telegram

        # Initialize components
        self.pandascore = PandaScoreFeed(api_key=settings.PANDASCORE_API_KEY)
        self.engine = PaperTradingEngine(
            model=model,
            capital=capital,
            min_edge=settings.MIN_EDGE_PCT,
        )

        # Track session data
        self.session_start: Optional[datetime] = None
        self.events_processed: int = 0
        self.trades_made: list[dict[str, Any]] = []
        self._running = False

    async def start(self) -> None:
        """Start the paper trading bot."""
        logger.info("starting_paper_trading_bot", game=self.game, capital=self.capital)

        self.session_start = datetime.now(timezone.utc)

        # Connect to PandaScore feed
        await self.pandascore.connect()

        # Register event handler
        self.pandascore.on_event(self._on_game_event)

        # Subscribe to live matches
        await self._subscribe_live_matches()

        self._running = True
        logger.info("paper_trading_bot_started")

        # Keep running and periodically refresh matches
        while self._running:
            await asyncio.sleep(60)
            await self._subscribe_live_matches()

    async def stop(self) -> None:
        """Stop the bot gracefully."""
        logger.info("stopping_paper_trading_bot")
        self._running = False

        await self.pandascore.disconnect()
        await self.engine.shutdown()

        logger.info("paper_trading_bot_stopped")

    async def _subscribe_live_matches(self) -> None:
        """Find and subscribe to currently live matches."""
        try:
            matches = await self.pandascore.get_live_matches(self.game)
        except Exception as e:
            logger.warning("get_live_matches_failed", error=str(e))
            return

        if not matches:
            logger.debug("no_live_matches", game=self.game)
            return

        for match in matches:
            match_id = str(match.get("id", ""))
            if not match_id:
                continue

            match_name = match.get("name", "Unknown")
            await self.pandascore.subscribe(self.game, match_id)
            logger.info("subscribed_to_match", match=match_name, match_id=match_id)

    async def _on_game_event(self, event: Any) -> None:
        """Handle incoming game event.

        Args:
            event: Game event from PandaScore feed
        """
        self.events_processed += 1

        # For paper trading, we need a market_id
        # In production, this would come from market mapping
        # For now, use match_id as placeholder
        market_id = f"paper_{event.match_id}"

        try:
            result = await self.engine.process_event(
                event=event,
                market_id=market_id,
                outcome="YES",
            )

            if result.get("trade"):
                trade = result["trade"]
                self.trades_made.append(trade)

                logger.info(
                    "paper_trade_executed",
                    event_type=event.event_type,
                    model_prediction=result["model_prediction"],
                    market_price=result["market_price"],
                    edge=result["edge"],
                    size=trade["size"],
                )

                if self.use_telegram:
                    await self._send_telegram_alert(result)

        except Exception as e:
            logger.error("event_processing_error", error=str(e), event_type=event.event_type)

    async def _send_telegram_alert(self, result: dict[str, Any]) -> None:
        """Send Telegram alert for a trade.

        Args:
            result: Trade result from engine
        """
        if not settings.TELEGRAM_BOT_TOKEN or not settings.TELEGRAM_CHAT_ID:
            logger.debug("telegram_not_configured")
            return

        # Build alert message
        trade = result["trade"]
        message = (
            f"Paper Trade Alert\n"
            f"Event: {trade['event_type']}\n"
            f"Model: {result['model_prediction']:.2%}\n"
            f"Market: {result['market_price']:.2%}\n"
            f"Edge: {result['edge']:.2%}\n"
            f"Size: ${trade['size']:.2f}"
        )

        try:
            import httpx

            async with httpx.AsyncClient() as client:
                await client.post(
                    f"https://api.telegram.org/bot{settings.TELEGRAM_BOT_TOKEN}/sendMessage",
                    json={
                        "chat_id": settings.TELEGRAM_CHAT_ID,
                        "text": message,
                    },
                )
        except Exception as e:
            logger.error("telegram_send_failed", error=str(e))

    def get_summary(self) -> dict[str, Any]:
        """Get session summary.

        Returns:
            Summary dictionary with session stats
        """
        session_end = datetime.now(timezone.utc)
        duration = (
            (session_end - self.session_start).total_seconds()
            if self.session_start
            else 0
        )

        total_edge = sum(t.get("edge", 0) for t in self.trades_made)
        total_size = sum(t.get("size", 0) for t in self.trades_made)

        return {
            "session_duration_seconds": duration,
            "events_processed": self.events_processed,
            "trades_made": len(self.trades_made),
            "total_edge": total_edge,
            "total_size": total_size,
            "avg_edge": total_edge / len(self.trades_made) if self.trades_made else 0,
        }


async def run_paper_trading(
    model_path: Path,
    game: str,
    capital: float,
    use_telegram: bool,
) -> None:
    """Run paper trading loop.

    Args:
        model_path: Path to trained model file
        game: Game to monitor
        capital: Starting capital
        use_telegram: Whether to send Telegram alerts
    """
    # Load model
    logger.info("loading_model", path=str(model_path))
    try:
        model = ImpactModel.load(model_path)
    except FileNotFoundError:
        logger.error("model_not_found", path=str(model_path))
        print(f"\nError: Model file not found: {model_path}")
        print("Train a model first with: uv run python scripts/train_model.py")
        sys.exit(1)
    except Exception as e:
        logger.error("model_load_error", error=str(e))
        print(f"\nError loading model: {e}")
        sys.exit(1)

    # Initialize bot
    bot = PaperTradingBot(
        model=model,
        game=game,
        capital=capital,
        use_telegram=use_telegram,
    )

    print(f"\n{'='*60}")
    print("PAPER TRADING STARTED")
    print(f"{'='*60}")
    print(f"Game:      {game}")
    print(f"Capital:   ${capital:,.2f}")
    print(f"Min Edge:  {settings.MIN_EDGE_PCT:.1%}")
    print(f"Model:     {model_path}")
    print(f"Telegram:  {'Enabled' if use_telegram else 'Disabled'}")
    print(f"{'='*60}")
    print("\nMonitoring for live events... (Press Ctrl+C to stop)\n")

    # Setup signal handlers
    loop = asyncio.get_running_loop()

    def signal_handler() -> None:
        logger.info("shutdown_signal_received")
        asyncio.create_task(bot.stop())

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    try:
        await bot.start()
    except KeyboardInterrupt:
        pass
    finally:
        await bot.stop()

        # Print summary
        summary = bot.get_summary()
        print(f"\n{'='*60}")
        print("PAPER TRADING SUMMARY")
        print(f"{'='*60}")
        print(f"Session Duration:  {summary['session_duration_seconds']:.0f}s")
        print(f"Events Processed:  {summary['events_processed']}")
        print(f"Trades Made:       {summary['trades_made']}")
        print(f"Total Size:        ${summary['total_size']:,.2f}")
        print(f"Total Edge:        {summary['total_edge']:.2%}")
        print(f"Avg Edge:          {summary['avg_edge']:.2%}")
        print(f"{'='*60}\n")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run paper trading with ML model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default=settings.ML_MODEL_PATH,
        help="Path to trained model file",
    )
    parser.add_argument(
        "--game",
        type=str,
        default="lol",
        choices=["lol", "csgo", "dota2", "valorant"],
        help="Game to monitor",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=settings.GLOBAL_CAPITAL,
        help="Starting capital for paper trading",
    )
    parser.add_argument(
        "--telegram",
        action="store_true",
        help="Enable Telegram alerts for trades",
    )

    args = parser.parse_args()

    asyncio.run(
        run_paper_trading(
            model_path=Path(args.model),
            game=args.game,
            capital=args.capital,
            use_telegram=args.telegram,
        )
    )


if __name__ == "__main__":
    main()
