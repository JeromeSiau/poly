"""Kalshi API executor implementing ExecutorProtocol.

Handles RSA-signed authentication, HTTP GET/POST, order placement and
cancellation against the Kalshi trading API.

Kalshi specifics:
- "ticker" instead of Polymarket's "token_id"
- Prices in cents (1-99) instead of 0.0-1.0
- Side is "yes"/"no" instead of "BUY"/"SELL"

The executor translates between the unified ExecutorProtocol interface
(0-1 prices, BUY/SELL) and Kalshi's native API conventions.
"""

from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import Any, Optional

import httpx
import structlog

from src.execution.models import OrderResult

logger = structlog.get_logger()

# Kalshi API endpoints
KALSHI_API_BASE = "https://api.elections.kalshi.com/trade-api/v2"
KALSHI_DEMO_API_BASE = "https://demo-api.kalshi.co/trade-api/v2"


class KalshiExecutor:
    """Kalshi exchange executor implementing ExecutorProtocol.

    Translates the unified interface to Kalshi's API:
    - token_id -> ticker
    - price (0.0-1.0) -> yes_price (cents 1-99)
    - side "BUY" -> action "buy", side "yes"
    - size (USD) -> count (contracts, at price in cents)
    """

    def __init__(
        self,
        api_base: str = KALSHI_API_BASE,
        api_key_id: str = "",
        private_key_pem: str = "",
    ) -> None:
        self.api_base = api_base
        self.api_key_id = api_key_id
        self.private_key_pem = private_key_pem
        self._client: Optional[httpx.AsyncClient] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            timeout = httpx.Timeout(20.0, connect=10.0)
            self._client = httpx.AsyncClient(timeout=timeout, http2=True)
        return self._client

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------

    def _auth_headers(self, method: str, path: str) -> dict[str, str]:
        """Generate RSA-signed auth headers for Kalshi API."""
        if not self.api_key_id or not self.private_key_pem:
            return {}

        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import padding

        timestamp_ms = str(int(time.time() * 1000))
        message = f"{timestamp_ms}{method.upper()}{path}"

        private_key = serialization.load_pem_private_key(
            self.private_key_pem.encode(), password=None
        )
        signature = private_key.sign(
            message.encode(),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=hashes.SHA256.digest_size,
            ),
            hashes.SHA256(),
        )

        return {
            "KALSHI-ACCESS-KEY": self.api_key_id,
            "KALSHI-ACCESS-TIMESTAMP": timestamp_ms,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode(),
        }

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    async def api_get(self, path: str, params: dict | None = None) -> Any:
        """GET request to Kalshi API with auth."""
        client = await self._ensure_client()
        url = f"{self.api_base}{path}"
        headers = self._auth_headers("GET", path)
        resp = await client.get(url, params=params, headers=headers)
        resp.raise_for_status()
        return resp.json()

    async def api_post(self, path: str, body: dict) -> Any:
        """POST request to Kalshi API with auth."""
        client = await self._ensure_client()
        url = f"{self.api_base}{path}"
        headers = self._auth_headers("POST", path)
        resp = await client.post(url, json=body, headers=headers)
        resp.raise_for_status()
        return resp.json()

    async def api_delete(self, path: str) -> Any:
        """DELETE request to Kalshi API with auth."""
        client = await self._ensure_client()
        url = f"{self.api_base}{path}"
        headers = self._auth_headers("DELETE", path)
        resp = await client.delete(url, headers=headers)
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # ExecutorProtocol
    # ------------------------------------------------------------------

    async def place_order(
        self,
        token_id: str,
        side: str,
        size: float,
        price: float,
        outcome: str = "",
        order_type: str = "GTC",
    ) -> OrderResult:
        """Place an order on Kalshi.

        Parameters follow ExecutorProtocol conventions:
        - token_id: Kalshi ticker (e.g. "KXBTCD-25FEB11-B98000")
        - side: "BUY" or "SELL"
        - size: dollar amount (converted to contract count)
        - price: 0.0-1.0 (converted to cents)
        - outcome: "yes" or "no" (default "yes")
        - order_type: "GTC" (default)
        """
        kalshi_outcome = (outcome or "yes").lower()
        price_cents = max(1, min(99, round(price * 100)))
        count = max(1, round(size / (price_cents / 100.0)))

        # Map unified side to Kalshi action
        action = "buy" if side.upper() == "BUY" else "sell"

        body: dict[str, Any] = {
            "ticker": token_id,
            "side": kalshi_outcome,
            "action": action,
            "type": "limit",
            f"{kalshi_outcome}_price": price_cents,
            "count": count,
        }

        # Map order_type to Kalshi time_in_force
        if order_type == "GTC":
            body["time_in_force"] = "good_till_canceled"
            body["post_only"] = True

        try:
            resp = await self.api_post("/portfolio/orders", body)
            order_id = resp.get("order", {}).get("order_id", "")
            if not order_id:
                return OrderResult(
                    order_id="",
                    filled=False,
                    status="error",
                    error=f"no order_id in response: {resp}",
                )
            return OrderResult(order_id=order_id, filled=False, status="placed")
        except Exception as exc:
            return OrderResult(
                order_id="",
                filled=False,
                status="error",
                error=str(exc),
            )

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order on Kalshi."""
        try:
            await self.api_delete(f"/portfolio/orders/{order_id}")
            return True
        except Exception as exc:
            logger.warning("kalshi_cancel_failed", order_id=order_id, error=str(exc))
            return False
