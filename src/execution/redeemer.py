"""Gas-free redemption of resolved Polymarket positions via Builder Relayer."""

from __future__ import annotations

import asyncio
from typing import Any

import structlog

from config.settings import settings

logger = structlog.get_logger()

try:
    from py_clob_client.client import ClobClient
    from py_builder_relayer_client.client import RelayClient
    from py_builder_signing_sdk.config import BuilderConfig
    from py_builder_signing_sdk.sdk_types import BuilderApiKeyCreds
    from poly_web3 import PolyWeb3Service, RELAYER_URL
except ImportError:
    ClobClient = None  # type: ignore[assignment,misc]
    RelayClient = None  # type: ignore[assignment,misc]
    BuilderConfig = None  # type: ignore[assignment,misc]
    BuilderApiKeyCreds = None  # type: ignore[assignment,misc]
    PolyWeb3Service = None  # type: ignore[assignment,misc]
    RELAYER_URL = ""


class PolymarketRedeemer:
    """Redeems resolved positions on Polymarket using the Builder Relayer (gas-free)."""

    def __init__(self, service: Any) -> None:
        self._service = service

    @classmethod
    def from_settings(cls) -> "PolymarketRedeemer":
        if PolyWeb3Service is None:
            raise ImportError("poly-web3 is not installed (pip install poly-web3)")

        if not settings.POLYMARKET_PRIVATE_KEY:
            raise ValueError("POLYMARKET_PRIVATE_KEY is required")
        if not settings.POLYMARKET_BUILDER_API_KEY:
            raise ValueError("POLYMARKET_BUILDER_API_KEY is required for redeems")

        clob = ClobClient(
            host=settings.POLYMARKET_CLOB_HTTP,
            chain_id=settings.POLYMARKET_CHAIN_ID,
            key=settings.POLYMARKET_PRIVATE_KEY,
            funder=settings.POLYMARKET_WALLET_ADDRESS,
            signature_type=1,  # POLY_PROXY
        )
        clob.set_api_creds(clob.create_or_derive_api_creds())

        relayer = RelayClient(
            relayer_url=RELAYER_URL,
            chain_id=settings.POLYMARKET_CHAIN_ID,
            private_key=settings.POLYMARKET_PRIVATE_KEY,
            builder_config=BuilderConfig(
                local_builder_creds=BuilderApiKeyCreds(
                    key=settings.POLYMARKET_BUILDER_API_KEY,
                    secret=settings.POLYMARKET_BUILDER_SECRET,
                    passphrase=settings.POLYMARKET_BUILDER_PASSPHRASE,
                )
            ),
        )

        service = PolyWeb3Service(
            clob_client=clob,
            relayer_client=relayer,
        )
        logger.info("redeemer_initialized", wallet=settings.POLYMARKET_WALLET_ADDRESS)
        return cls(service)

    def _redeem_all_sync(self, batch_size: int = 10) -> list[dict[str, Any]]:
        results = self._service.redeem_all(batch_size=batch_size)
        out: list[dict[str, Any]] = []
        for r in results:
            if r is None:
                out.append({"status": "failed"})
            elif isinstance(r, dict):
                out.append(r)
            else:
                out.append({"status": "ok", "result": r})
        return out

    async def redeem_all(self, batch_size: int = 10) -> list[dict[str, Any]]:
        """Redeem all redeemable positions. Returns list of results."""
        return await asyncio.to_thread(self._redeem_all_sync, batch_size)

    def _redeem_sync(self, condition_ids: list[str], batch_size: int = 10) -> list[dict[str, Any]]:
        results = self._service.redeem(condition_ids, batch_size=batch_size)
        out: list[dict[str, Any]] = []
        for r in (results if isinstance(results, list) else [results]):
            if r is None:
                out.append({"status": "failed"})
            elif isinstance(r, dict):
                out.append(r)
            else:
                out.append({"status": "ok", "result": r})
        return out

    async def redeem(self, condition_ids: list[str], batch_size: int = 10) -> list[dict[str, Any]]:
        """Redeem specific condition IDs."""
        return await asyncio.to_thread(self._redeem_sync, condition_ids, batch_size)

    def is_resolved(self, condition_id: str) -> bool:
        """Check if a condition is resolved on-chain."""
        return self._service.is_condition_resolved(condition_id)
