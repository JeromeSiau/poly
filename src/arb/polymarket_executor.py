"""Polymarket executor using py-clob-client."""

from __future__ import annotations

import asyncio
from typing import Optional, Any

import structlog

from config.settings import settings

logger = structlog.get_logger()

try:
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import ApiCreds, OrderArgs, OrderType
except ImportError:  # pragma: no cover - optional dependency in runtime
    ClobClient = None
    ApiCreds = None
    OrderArgs = None
    OrderType = None


class PolymarketExecutor:
    """Executes orders on Polymarket via the CLOB API."""

    def __init__(
        self,
        host: str,
        chain_id: int,
        private_key: str,
        funder: str,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        api_passphrase: Optional[str] = None,
    ) -> None:
        if ClobClient is None:
            raise ImportError("py_clob_client is not installed")

        if not private_key:
            raise ValueError("POLYMARKET_PRIVATE_KEY is required for execution")

        self._client = ClobClient(
            host=host,
            chain_id=chain_id,
            key=private_key,
            funder=funder,
            signature_type=1,  # POLY_PROXY: EOA signs for proxy wallet
        )
        self._api_key = api_key
        self._api_secret = api_secret
        self._api_passphrase = api_passphrase
        self._creds_ready = False

    @classmethod
    def from_settings(cls) -> "PolymarketExecutor":
        """Build executor from global settings.

        Raises ConfigError if credentials are missing.
        """
        from config.settings import settings
        from config.validators import validate_polymarket_credentials
        validate_polymarket_credentials()
        return cls(
            host=settings.POLYMARKET_CLOB_HTTP,
            chain_id=settings.POLYMARKET_CHAIN_ID,
            private_key=settings.POLYMARKET_PRIVATE_KEY,
            funder=settings.POLYMARKET_WALLET_ADDRESS,
            api_key=settings.POLYMARKET_API_KEY or None,
            api_secret=settings.POLYMARKET_API_SECRET or None,
            api_passphrase=settings.POLYMARKET_API_PASSPHRASE or None,
        )

    def _ensure_creds(self) -> None:
        if self._creds_ready:
            return

        creds: Optional[ApiCreds]
        if self._api_key and self._api_secret and self._api_passphrase:
            creds = ApiCreds(
                api_key=self._api_key,
                api_secret=self._api_secret,
                api_passphrase=self._api_passphrase,
            )
        else:
            creds = self._client.create_or_derive_api_creds()

        if not creds:
            raise RuntimeError("Failed to create or derive Polymarket API credentials")

        self._client.set_api_creds(creds)
        self._creds_ready = True

    @staticmethod
    def _resolve_order_type(order_type: str) -> Any:
        if OrderType is None:
            return None

        try:
            return getattr(OrderType, order_type.upper())
        except AttributeError:
            return OrderType.FOK

    @staticmethod
    def _normalize_response(response: Any) -> dict[str, Any]:
        if isinstance(response, dict):
            if response.get("error") or response.get("errorMessage"):
                return {"status": "ERROR", "error": response}
            if response.get("status"):
                return response
            return {"status": "PLACED", **response}
        return {"status": "PLACED", "response": response}

    def _place_order_sync(
        self,
        token_id: str,
        side: str,
        size: float,
        price: float,
        outcome: Optional[str] = None,
        order_type: str = "",
        force_taker: bool = False,
    ) -> dict[str, Any]:
        self._ensure_creds()

        side = side.upper()
        if side not in ("BUY", "SELL"):
            return {"status": "ERROR", "message": f"Invalid side: {side}"}

        if price <= 0:
            return {"status": "ERROR", "message": "Invalid price"}

        # Convert USD size into shares.
        # BUY: cost per share = price → shares = size / price
        # SELL: collateral per share = (1 - price) → shares = size / (1 - price)
        if side == "BUY":
            shares = size / price
        else:
            shares = size / (1.0 - price)
        if shares <= 0:
            return {"status": "ERROR", "message": "Invalid size"}

        order_args = OrderArgs(
            token_id=token_id,
            price=price,
            size=shares,
            side=side,
            fee_rate_bps=settings.POLYMARKET_FEE_BPS,
        )

        order = self._client.create_order(order_args)
        resolved_type = self._resolve_order_type(
            order_type if order_type else settings.POLYMARKET_ORDER_TYPE
        )
        # post_only is only valid for GTC/GTD — never for FOK/IOC.
        # force_taker=True disables post_only (used by stoploss sells).
        if force_taker:
            use_post_only = False
        else:
            use_post_only = settings.POLYMARKET_POST_ONLY and resolved_type in ("GTC", "GTD")
        response = self._client.post_order(
            order,
            orderType=resolved_type,
            post_only=use_post_only,
        )

        logger.info(
            "polymarket_order_posted",
            token_id=token_id,
            side=side,
            outcome=outcome,
            size=size,
            price=price,
            order_type=order_type or settings.POLYMARKET_ORDER_TYPE,
        )

        return self._normalize_response(response)

    async def place_order(
        self,
        token_id: str,
        side: str,
        size: float,
        price: float,
        outcome: Optional[str] = None,
        order_type: str = "",
        force_taker: bool = False,
    ) -> dict[str, Any]:
        """Place an order asynchronously.

        Args:
            order_type: Override order type (e.g. "GTC", "FOK").
                        Empty string falls back to settings.POLYMARKET_ORDER_TYPE.
            force_taker: If True, disables post_only regardless of settings.
        """
        try:
            return await asyncio.to_thread(
                self._place_order_sync,
                token_id,
                side,
                size,
                price,
                outcome,
                order_type,
                force_taker,
            )
        except Exception as e:
            logger.error("polymarket_order_failed", error=str(e))
            return {"status": "ERROR", "message": str(e)}

    def _cancel_order_sync(self, order_id: str) -> dict[str, Any]:
        self._ensure_creds()
        try:
            response = self._client.cancel(order_id)
        except Exception as exc:
            return {"status": "ERROR", "message": str(exc)}
        return self._normalize_response(response)

    async def cancel_order(self, order_id: str) -> dict[str, Any]:
        """Cancel a live order by ID."""
        try:
            return await asyncio.to_thread(self._cancel_order_sync, order_id)
        except Exception as e:
            logger.error("polymarket_cancel_failed", order_id=order_id, error=str(e))
            return {"status": "ERROR", "message": str(e)}

    def _cancel_all_sync(self) -> list:
        self._ensure_creds()
        try:
            return self._client.cancel_all()
        except Exception as exc:
            logger.warning("polymarket_cancel_all_failed", error=str(exc))
            return []

    async def cancel_all_orders(self) -> list:
        """Cancel ALL open orders on the CLOB for this wallet."""
        return await asyncio.to_thread(self._cancel_all_sync)

    def _get_open_orders_sync(self, market: str = "") -> list[dict[str, Any]]:
        self._ensure_creds()
        try:
            params: dict[str, Any] = {}
            if market:
                params["market"] = market
            orders = self._client.get_orders(**params)
            return orders if isinstance(orders, list) else []
        except Exception as exc:
            logger.warning("polymarket_get_orders_failed", error=str(exc))
            return []

    async def get_open_orders(self, market: str = "") -> list[dict[str, Any]]:
        """Get open/live orders, optionally filtered by market (condition_id)."""
        return await asyncio.to_thread(self._get_open_orders_sync, market)

    def _get_order_sync(self, order_id: str) -> dict[str, Any] | None:
        self._ensure_creds()
        try:
            return self._client.get_order(order_id)
        except Exception as exc:
            logger.warning("polymarket_get_order_failed", order_id=order_id[:16], error=str(exc))
            return None

    async def get_order(self, order_id: str) -> dict[str, Any] | None:
        """Fetch a single order by ID (any status: LIVE, MATCHED, CANCELLED)."""
        return await asyncio.to_thread(self._get_order_sync, order_id)

    # USDC.e on Polygon (PoS-bridged, 6 decimals) — used by Polymarket
    _USDC_E_ADDRESS = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
    _BALANCE_OF_SELECTOR = "0x70a08231"
    _DEFAULT_RPC = "https://polygon-rpc.com"

    def _get_balance_sync(self) -> float:
        """Query on-chain USDC.e balance on Polygon for the funder wallet."""
        import httpx

        rpc_url = settings.POLYGON_RPC_URL or self._DEFAULT_RPC
        wallet = settings.POLYMARKET_WALLET_ADDRESS
        padded = wallet.lower().replace("0x", "").zfill(64)
        data = self._BALANCE_OF_SELECTOR + padded

        try:
            resp = httpx.post(
                rpc_url,
                json={
                    "jsonrpc": "2.0",
                    "method": "eth_call",
                    "params": [{"to": self._USDC_E_ADDRESS, "data": data}, "latest"],
                    "id": 1,
                },
                timeout=10,
            )
            result = resp.json().get("result", "0x0")
            return int(result, 16) / 1e6
        except Exception as exc:
            logger.warning("polymarket_get_balance_failed", error=str(exc))
            return 0.0

    async def get_balance(self) -> float:
        """Return on-chain USDC.e balance for the wallet on Polygon."""
        return await asyncio.to_thread(self._get_balance_sync)
