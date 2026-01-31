# src/bot/crossmarket_handlers.py
"""Telegram handlers for cross-market arbitrage alerts.

This module provides handlers for sending cross-market arbitrage alerts to
Telegram users, with inline keyboard buttons for approving or skipping trades.
"""

import asyncio
import time
from typing import Any, Optional

import structlog

from config.settings import settings
from src.arb.cross_market_arb import CrossMarketOpportunity

logger = structlog.get_logger()


class CrossMarketArbHandler:
    """Telegram handler for cross-market arbitrage alerts.

    Sends alerts when cross-market arbitrage opportunities are detected and
    handles user interactions via inline keyboard buttons.

    Attributes:
        bot: Telegram bot instance.
        chat_id: Telegram chat ID to send alerts to.
        alert_expiry_seconds: Time in seconds before alerts expire.
    """

    def __init__(
        self,
        bot: Any,
        chat_id: str,
        alert_expiry_seconds: Optional[int] = None,
    ) -> None:
        """Initialize the handler.

        Args:
            bot: Telegram bot instance.
            chat_id: Telegram chat ID to send alerts to.
            alert_expiry_seconds: Override for alert expiry time (default from settings).
        """
        self.bot = bot
        self.chat_id = chat_id
        self.alert_expiry_seconds = (
            alert_expiry_seconds
            if alert_expiry_seconds is not None
            else settings.CROSSMARKET_ALERT_EXPIRY_SECONDS
        )

        # Pending opportunities: message_id -> (opportunity, position_size, timestamp)
        self._pending: dict[str, tuple[CrossMarketOpportunity, float, float]] = {}
        self._expiry_tasks: dict[str, asyncio.Task] = {}

    def format_alert_message(
        self,
        opportunity: CrossMarketOpportunity,
        position_size: float,
    ) -> str:
        """Format a cross-market opportunity as a Telegram alert message.

        Args:
            opportunity: The cross-market arbitrage opportunity to format.
            position_size: Suggested position size in USD.

        Returns:
            Formatted alert message string with table layout.
        """
        # Get event info
        event_name = getattr(opportunity.event, "name", str(opportunity.event))
        confidence = getattr(opportunity.event, "confidence", 0.0)
        confidence_pct = confidence * 100 if confidence <= 1 else confidence

        # Format prices as cents
        source_price_cents = int(opportunity.source_price * 100)
        target_price_cents = int(opportunity.target_price * 100)

        # Format liquidity
        source_liq = f"${opportunity.source_liquidity:,.0f}"
        target_liq = f"${opportunity.target_liquidity:,.0f}"

        # Calculate expected profit
        net_edge_pct = opportunity.net_edge_pct
        expected_profit = position_size * net_edge_pct

        # Platform names (capitalize for display)
        source_name = opportunity.source_platform.capitalize()
        target_name = opportunity.target_platform.capitalize()

        # Pad platform names for alignment (max 10 chars)
        src_col = source_name[:10].center(10)
        tgt_col = target_name[:10].center(10)

        # Build the alert message
        alert = (
            f"[TARGET] CROSS-MARKET ARB\n"
            f"\n"
            f"Event: {event_name}\n"
            f"Match confidence: {confidence_pct:.0f}%\n"
            f"\n"
            f"+-------------+------------+------------+\n"
            f"|             |{src_col}|{tgt_col}|\n"
            f"+-------------+------------+------------+\n"
            f"| YES price   |    {source_price_cents:>3}c    |    {target_price_cents:>3}c    |\n"
            f"| Liquidity   | {source_liq:>10} | {target_liq:>10} |\n"
            f"+-------------+------------+------------+\n"
            f"\n"
            f"Position: ${position_size:,.0f} | Net edge: {net_edge_pct * 100:.1f}%\n"
            f"Expected profit: ~${expected_profit:,.2f}\n"
            f"\n"
            f"Expires in {self.alert_expiry_seconds}s"
        )

        return alert

    def get_approval_keyboard(self, message_id: str) -> Any:
        """Create an inline keyboard with approve/skip buttons.

        Args:
            message_id: The message ID for callback data.

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
                        {
                            "text": "APPROVE",
                            "callback_data": f"crossmarket_approve:{message_id}",
                        },
                        {
                            "text": "SKIP",
                            "callback_data": f"crossmarket_skip:{message_id}",
                        },
                    ]
                ]
            }

        keyboard = [
            [
                InlineKeyboardButton(
                    "APPROVE",
                    callback_data=f"crossmarket_approve:{message_id}",
                ),
                InlineKeyboardButton(
                    "SKIP",
                    callback_data=f"crossmarket_skip:{message_id}",
                ),
            ]
        ]

        return InlineKeyboardMarkup(keyboard)

    async def send_alert(
        self,
        opportunity: CrossMarketOpportunity,
        position_size: float,
    ) -> Optional[int]:
        """Send an alert to the Telegram chat.

        Args:
            opportunity: The cross-market arbitrage opportunity.
            position_size: Suggested position size in USD.

        Returns:
            Message ID if successful, None otherwise.
        """
        alert_text = self.format_alert_message(opportunity, position_size)

        try:
            # Send message first to get message_id
            message = await self.bot.send_message(
                chat_id=self.chat_id,
                text=alert_text,
                reply_markup=self.get_approval_keyboard("pending"),
            )

            message_id = message.message_id
            message_id_str = str(message_id)

            # Update keyboard with actual message_id
            keyboard = self.get_approval_keyboard(message_id_str)

            # Store pending opportunity
            self._pending[message_id_str] = (opportunity, position_size, time.time())

            # Try to edit message with correct callback data
            try:
                await self.bot.edit_message_reply_markup(
                    chat_id=self.chat_id,
                    message_id=message_id,
                    reply_markup=keyboard,
                )
            except Exception:
                # If edit fails, the pending message_id still works
                pass

            # Set up expiry task
            expiry_task = asyncio.create_task(
                self._expire_opportunity(message_id_str, message_id)
            )
            self._expiry_tasks[message_id_str] = expiry_task

            logger.info(
                "crossmarket_alert_sent",
                message_id=message_id,
                chat_id=self.chat_id,
                source_platform=opportunity.source_platform,
                target_platform=opportunity.target_platform,
            )

            return message_id

        except Exception as e:
            logger.error(
                "crossmarket_alert_send_failed",
                error=str(e),
                event_name=getattr(opportunity.event, "name", "unknown"),
            )
            return None

    async def _expire_opportunity(
        self,
        message_id_str: str,
        message_id: int,
    ) -> None:
        """Expire an opportunity after the timeout period.

        Args:
            message_id_str: String message ID key for pending dict.
            message_id: Integer message ID for Telegram API.
        """
        await asyncio.sleep(self.alert_expiry_seconds)

        if message_id_str in self._pending:
            self._pending.pop(message_id_str, None)
            self._expiry_tasks.pop(message_id_str, None)

            try:
                await self.bot.edit_message_text(
                    chat_id=self.chat_id,
                    message_id=message_id,
                    text="[EXPIRED] This cross-market opportunity has expired.",
                )
            except Exception as e:
                logger.warning(
                    "crossmarket_expire_message_update_failed",
                    error=str(e),
                    message_id=message_id,
                )

            logger.info("crossmarket_opportunity_expired", message_id=message_id)

    def add_pending_opportunity(
        self,
        message_id: str,
        opportunity: Any,
        position_size: float = 0.0,
    ) -> None:
        """Add a pending opportunity.

        Args:
            message_id: The message ID key.
            opportunity: The opportunity to store.
            position_size: The position size in USD.
        """
        self._pending[message_id] = (opportunity, position_size, time.time())

    def get_pending_opportunity(self, message_id: str) -> Optional[Any]:
        """Get a pending opportunity by message ID.

        Args:
            message_id: The message ID key.

        Returns:
            The opportunity if found, None otherwise.
        """
        entry = self._pending.get(message_id)
        if entry:
            return entry[0]
        return None

    def remove_pending_opportunity(self, message_id: str) -> None:
        """Remove a pending opportunity.

        Args:
            message_id: The message ID key to remove.
        """
        self._pending.pop(message_id, None)

        # Cancel expiry task if exists
        if message_id in self._expiry_tasks:
            self._expiry_tasks[message_id].cancel()
            self._expiry_tasks.pop(message_id, None)

    async def handle_approve(self, message_id: str) -> dict[str, Any]:
        """Handle approval of a cross-market opportunity.

        Args:
            message_id: The message ID of the opportunity to approve.

        Returns:
            Result dictionary with approval status.
        """
        if message_id not in self._pending:
            return {
                "approved": False,
                "error": "Opportunity not found or expired",
            }

        opportunity, position_size, timestamp = self._pending.pop(message_id)

        # Cancel expiry task
        if message_id in self._expiry_tasks:
            self._expiry_tasks[message_id].cancel()
            self._expiry_tasks.pop(message_id, None)

        logger.info(
            "crossmarket_trade_approved",
            message_id=message_id,
            source_platform=opportunity.source_platform,
            target_platform=opportunity.target_platform,
            position_size=position_size,
        )

        return {
            "approved": True,
            "opportunity": opportunity,
            "position_size": position_size,
        }

    async def handle_skip(self, message_id: str) -> dict[str, Any]:
        """Handle skipping of a cross-market opportunity.

        Args:
            message_id: The message ID of the opportunity to skip.

        Returns:
            Result dictionary with skip status.
        """
        if message_id not in self._pending:
            return {
                "skipped": False,
                "error": "Opportunity not found or expired",
            }

        opportunity, position_size, timestamp = self._pending.pop(message_id)

        # Cancel expiry task
        if message_id in self._expiry_tasks:
            self._expiry_tasks[message_id].cancel()
            self._expiry_tasks.pop(message_id, None)

        logger.info("crossmarket_trade_skipped", message_id=message_id)

        return {"skipped": True}

    def cleanup_expired_opportunities(self) -> int:
        """Clean up expired opportunities from the pending dict.

        Returns:
            Number of opportunities cleaned up.
        """
        current_time = time.time()
        expired_keys = []

        for message_id, (opp, size, timestamp) in self._pending.items():
            if current_time - timestamp > self.alert_expiry_seconds:
                expired_keys.append(message_id)

        for key in expired_keys:
            self._pending.pop(key, None)
            if key in self._expiry_tasks:
                self._expiry_tasks[key].cancel()
                self._expiry_tasks.pop(key, None)

        if expired_keys:
            logger.info(
                "crossmarket_opportunities_cleaned_up",
                count=len(expired_keys),
            )

        return len(expired_keys)

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

        action, message_id = callback_data.split(":", 1)

        if action == "crossmarket_approve":
            result = await self.handle_approve(message_id)
            if result.get("approved"):
                await query.edit_message_text(
                    "[APPROVED] Cross-market trade approved.\n"
                    "Execution pending..."
                )
            else:
                await query.edit_message_text(
                    f"[ERROR] Could not approve: {result.get('error', 'Unknown error')}"
                )

        elif action == "crossmarket_skip":
            result = await self.handle_skip(message_id)
            if result.get("skipped"):
                await query.edit_message_text("[SKIPPED] Opportunity skipped.")
            else:
                await query.edit_message_text(
                    f"[ERROR] Skip failed: {result.get('error', 'Unknown error')}"
                )
