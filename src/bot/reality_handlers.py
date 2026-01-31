# src/bot/reality_handlers.py
"""Telegram handlers for reality arbitrage alerts.

This module provides handlers for sending trading alerts to Telegram users,
with inline keyboard buttons for approving or skipping trades.
"""

import asyncio
from typing import Any, Optional

import structlog

from src.arb.reality_arb import ArbOpportunity

logger = structlog.get_logger()


class RealityArbHandler:
    """Telegram handler for reality arbitrage alerts.

    Sends alerts when arbitrage opportunities are detected and handles
    user interactions via inline keyboard buttons.

    Attributes:
        ALERT_EXPIRY_SECONDS: Default expiry time for alerts in seconds.
    """

    ALERT_EXPIRY_SECONDS = 30

    def __init__(self, bot: Any, engine: Any) -> None:
        """Initialize the handler.

        Args:
            bot: Telegram bot instance.
            engine: RealityArbEngine instance for trade execution.
        """
        self.bot = bot
        self.engine = engine
        self._pending: dict[str, tuple[ArbOpportunity, float]] = {}
        self._expiry_tasks: dict[str, asyncio.Task] = {}

    def format_alert(
        self,
        opportunity: ArbOpportunity,
        market_title: str,
        expiry_seconds: Optional[int] = None,
    ) -> str:
        """Format an arbitrage opportunity as a Telegram alert message.

        Args:
            opportunity: The arbitrage opportunity to format.
            market_title: Human-readable market title.
            expiry_seconds: Expiry time in seconds (optional).

        Returns:
            Formatted alert message string.
        """
        expiry = expiry_seconds or self.ALERT_EXPIRY_SECONDS
        edge_display = f"{opportunity.edge_pct * 100:.0f}%"

        alert = (
            f"REALITY ARB ALERT\n"
            f"=================\n"
            f"\n"
            f"Market: {market_title}\n"
            f"Action: {opportunity.side} {opportunity.outcome}\n"
            f"Edge: {edge_display}\n"
            f"\n"
            f"Current: {opportunity.current_price:.2f}\n"
            f"Fair: {opportunity.estimated_fair_price:.2f}\n"
            f"\n"
            f"Trigger: {opportunity.trigger_event}\n"
            f"\n"
            f"Expires in {expiry}s"
        )

        return alert

    def get_approval_keyboard(self, market_id: str) -> Any:
        """Create an inline keyboard with approve/skip buttons.

        Args:
            market_id: The market ID for callback data.

        Returns:
            InlineKeyboardMarkup with approve and skip buttons.
        """
        try:
            from telegram import InlineKeyboardButton, InlineKeyboardMarkup
        except ImportError:
            # Return a simple dict representation if telegram is not installed
            return {
                "inline_keyboard": [
                    [
                        {"text": "Approve", "callback_data": f"approve:{market_id}"},
                        {"text": "Skip", "callback_data": f"skip:{market_id}"},
                    ]
                ]
            }

        keyboard = [
            [
                InlineKeyboardButton(
                    "Approve", callback_data=f"approve:{market_id}"
                ),
                InlineKeyboardButton("Skip", callback_data=f"skip:{market_id}"),
            ]
        ]

        return InlineKeyboardMarkup(keyboard)

    async def send_alert(
        self,
        chat_id: int,
        opportunity: ArbOpportunity,
        market_title: str,
        size: float,
    ) -> Optional[int]:
        """Send an alert to a Telegram chat.

        Args:
            chat_id: Telegram chat ID to send the alert to.
            opportunity: The arbitrage opportunity.
            market_title: Human-readable market title.
            size: Suggested position size.

        Returns:
            Message ID if successful, None otherwise.
        """
        alert_text = self.format_alert(opportunity, market_title)
        keyboard = self.get_approval_keyboard(opportunity.market_id)

        try:
            message = await self.bot.send_message(
                chat_id=chat_id,
                text=alert_text,
                reply_markup=keyboard,
            )

            message_id = message.message_id

            # Store pending opportunity
            self._pending[opportunity.market_id] = (opportunity, size)

            # Set up expiry task
            expiry_task = asyncio.create_task(
                self._expire_opportunity(opportunity.market_id, chat_id, message_id)
            )
            self._expiry_tasks[opportunity.market_id] = expiry_task

            logger.info(
                "alert_sent",
                market_id=opportunity.market_id,
                chat_id=chat_id,
                message_id=message_id,
            )

            return message_id

        except Exception as e:
            logger.error(
                "alert_send_failed",
                error=str(e),
                market_id=opportunity.market_id,
            )
            return None

    async def _expire_opportunity(
        self,
        market_id: str,
        chat_id: int,
        message_id: int,
    ) -> None:
        """Expire an opportunity after the timeout period.

        Args:
            market_id: The market ID of the opportunity.
            chat_id: Telegram chat ID.
            message_id: Message ID to update.
        """
        await asyncio.sleep(self.ALERT_EXPIRY_SECONDS)

        if market_id in self._pending:
            self._pending.pop(market_id, None)
            self._expiry_tasks.pop(market_id, None)

            try:
                await self.bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=message_id,
                    text="[EXPIRED] This opportunity has expired.",
                )
            except Exception as e:
                logger.warning(
                    "expire_message_update_failed",
                    error=str(e),
                    market_id=market_id,
                )

            logger.info("opportunity_expired", market_id=market_id)

    async def handle_approve(self, market_id: str) -> dict[str, Any]:
        """Handle approval of a trading opportunity.

        Args:
            market_id: The market ID to approve.

        Returns:
            Result dictionary with execution status.
        """
        if market_id not in self._pending:
            return {"executed": False, "error": "Opportunity not found or expired"}

        opportunity, size = self._pending.pop(market_id)

        # Cancel expiry task
        if market_id in self._expiry_tasks:
            self._expiry_tasks[market_id].cancel()
            self._expiry_tasks.pop(market_id, None)

        try:
            result = await self.engine.execute(opportunity, size)

            logger.info(
                "trade_approved",
                market_id=market_id,
                result=result,
            )

            return {"executed": True, "result": result}

        except Exception as e:
            logger.error(
                "trade_execution_failed",
                error=str(e),
                market_id=market_id,
            )
            return {"executed": False, "error": str(e)}

    async def handle_skip(self, market_id: str) -> dict[str, Any]:
        """Handle skipping of a trading opportunity.

        Args:
            market_id: The market ID to skip.

        Returns:
            Result dictionary with skip status.
        """
        if market_id not in self._pending:
            return {"skipped": False, "error": "Opportunity not found or expired"}

        self._pending.pop(market_id, None)

        # Cancel expiry task
        if market_id in self._expiry_tasks:
            self._expiry_tasks[market_id].cancel()
            self._expiry_tasks.pop(market_id, None)

        logger.info("trade_skipped", market_id=market_id)

        return {"skipped": True}

    async def callback_handler(self, update: Any, context: Any) -> None:
        """Handle callback queries from inline keyboard buttons.

        Args:
            update: Telegram Update object.
            context: Telegram Context object.
        """
        query = update.callback_query

        if not query or not query.data:
            return

        await query.answer()

        callback_data = query.data
        if ":" not in callback_data:
            return

        action, market_id = callback_data.split(":", 1)

        if action == "approve":
            result = await self.handle_approve(market_id)
            if result.get("executed"):
                await query.edit_message_text(
                    f"[EXECUTED] Trade approved and executed.\n"
                    f"Status: {result.get('result', {}).get('status', 'UNKNOWN')}"
                )
            else:
                await query.edit_message_text(
                    f"[ERROR] Execution failed: {result.get('error', 'Unknown error')}"
                )

        elif action == "skip":
            result = await self.handle_skip(market_id)
            if result.get("skipped"):
                await query.edit_message_text("[SKIPPED] Opportunity skipped.")
            else:
                await query.edit_message_text(
                    f"[ERROR] Skip failed: {result.get('error', 'Unknown error')}"
                )
