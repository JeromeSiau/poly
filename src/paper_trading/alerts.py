"""Telegram alerts for paper trading opportunities."""

import json
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Optional

import structlog

from config.settings import settings

logger = structlog.get_logger()

# Telegram API base URL
TELEGRAM_API_URL = "https://api.telegram.org/bot{token}/sendMessage"


@dataclass
class TelegramAlerter:
    """Send alerts to Telegram.

    Uses httpx for async HTTP requests when available,
    falls back to urllib for sync requests otherwise.
    """

    bot_token: str = ""
    chat_id: str = ""
    _httpx_client: Optional[object] = field(default=None, repr=False)

    def __post_init__(self):
        self.bot_token = self.bot_token or settings.TELEGRAM_BOT_TOKEN
        self.chat_id = self.chat_id or settings.TELEGRAM_CHAT_ID

    async def _get_httpx_client(self):
        """Lazy initialize httpx client."""
        if self._httpx_client is None:
            try:
                import httpx

                self._httpx_client = httpx.AsyncClient(timeout=30.0)
            except ImportError:
                logger.debug("httpx_not_available", msg="Using urllib fallback")
                self._httpx_client = False  # Mark as unavailable
        return self._httpx_client

    async def send_opportunity_alert(
        self,
        match_name: str,
        event_type: str,
        team: str,
        game_time: float,
        gold_diff: int,
        model_prediction: float,
        market_price: float,
        edge: float,
        trade_size: float,
    ) -> bool:
        """Send opportunity detected alert."""
        confidence = "HIGH" if edge > 0.10 else "MEDIUM"
        message = f"""
*OPPORTUNITY DETECTED*

*Match:* {match_name}
*Event:* {event_type} by {team}
*Game:* {game_time:.0f}min | Gold: {gold_diff:+,}

*Model:* {model_prediction:.0%} -> *Market:* {market_price:.0%}
*Edge:* +{edge:.1%} | Confidence: {confidence}

*Simulated trade:*
   BUY {team} @ {market_price:.1%}
   Size: ${trade_size:.2f}

Will update at T+30s, T+60s, T+120s
"""
        return await self._send(message)

    async def send_followup_alert(
        self,
        match_name: str,
        team: str,
        initial_price: float,
        current_price: float,
        time_elapsed: int,
        edge_captured: float,
        running_pnl: float,
    ) -> bool:
        """Send follow-up price update alert."""
        direction = "UP" if current_price > initial_price else "DOWN"
        check = "OK" if edge_captured > 0 else "X"
        message = f"""
*UPDATE: {match_name}* [{direction}]

T+{time_elapsed}s: Market moved {initial_price:.0%} -> {current_price:.0%}
Edge captured: {edge_captured:+.1%} [{check}]
Running P&L: ${running_pnl:+.2f}
"""
        return await self._send(message)

    async def send_match_result(
        self,
        match_name: str,
        winner: str,
        our_bet: str,
        pnl: float,
        total_pnl: float,
    ) -> bool:
        """Send match result alert."""
        won = winner == our_bet
        result_text = "WIN" if won else "LOSS"
        message = f"""
*MATCH RESULT: {match_name}*

Winner: {winner}
Our bet: {our_bet}
Result: {result_text}

P&L: ${pnl:+.2f}
Session Total: ${total_pnl:+.2f}
"""
        return await self._send(message)

    async def send_custom_alert(self, message: str) -> bool:
        """Send a custom message."""
        return await self._send(message)

    async def _send(self, message: str) -> bool:
        """Send message to Telegram."""
        if not self.bot_token or not self.chat_id:
            logger.warning("telegram_not_configured")
            return False

        try:
            client = await self._get_httpx_client()

            if client and client is not False:
                # Use httpx async client
                return await self._send_httpx(client, message)
            else:
                # Fallback to urllib (sync, run in thread)
                return await self._send_urllib(message)

        except Exception as e:
            logger.error("telegram_send_failed", error=str(e))
            return False

    async def _send_httpx(self, client, message: str) -> bool:
        """Send message using httpx."""
        url = TELEGRAM_API_URL.format(token=self.bot_token)
        payload = {
            "chat_id": self.chat_id,
            "text": message.strip(),
            "parse_mode": "Markdown",
        }

        response = await client.post(url, json=payload)
        if response.status_code == 200:
            logger.debug("telegram_sent", chat_id=self.chat_id)
            return True
        else:
            logger.error(
                "telegram_api_error",
                status=response.status_code,
                response=response.text,
            )
            return False

    async def _send_urllib(self, message: str) -> bool:
        """Send message using urllib (sync fallback)."""
        import asyncio

        # Run sync code in thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._send_urllib_sync, message)

    def _send_urllib_sync(self, message: str) -> bool:
        """Synchronous urllib implementation."""
        url = TELEGRAM_API_URL.format(token=self.bot_token)
        payload = {
            "chat_id": self.chat_id,
            "text": message.strip(),
            "parse_mode": "Markdown",
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=30) as response:
            if response.status == 200:
                logger.debug("telegram_sent", chat_id=self.chat_id)
                return True
            else:
                logger.error("telegram_api_error", status=response.status)
                return False

    async def close(self):
        """Close the httpx client if open."""
        if self._httpx_client and self._httpx_client is not False:
            await self._httpx_client.aclose()
            self._httpx_client = None
