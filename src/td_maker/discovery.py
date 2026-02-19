# src/td_maker/discovery.py
from __future__ import annotations
import time
from typing import Any
import structlog
from src.td_maker.state import MarketState, MarketRegistry
from src.td_maker.resilience import clob_retry
from src.utils.crypto_markets import fetch_crypto_markets, SLUG_TO_CHAINLINK

logger = structlog.get_logger()


def parse_slug_info(slug: str, slot_duration: int = 15 * 60) -> tuple[str, int, int]:
    """Parse 'btc-up-15m-1771079400' -> (chainlink_symbol, slot_ts, slot_duration).

    slot_duration is inferred from slug if it contains '5m' or '15m',
    otherwise falls back to the passed default.
    """
    parts = slug.split("-")
    try:
        slot_ts = int(parts[-1])
    except (ValueError, IndexError):
        slot_ts = 0
    symbol_key = parts[0] if parts else ""
    chainlink_symbol = SLUG_TO_CHAINLINK.get(symbol_key, f"{symbol_key}/usd")

    # Infer slot duration from slug
    if "5m" in parts and "15m" not in parts:
        inferred_duration = 5 * 60
    elif "15m" in parts:
        inferred_duration = 15 * 60
    else:
        inferred_duration = slot_duration

    return chainlink_symbol, slot_ts, inferred_duration


class MarketDiscovery:

    def __init__(self, poly_feed: Any, chainlink_feed: Any, config: Any) -> None:
        self.poly_feed = poly_feed
        self.chainlink = chainlink_feed
        self.config = config

    async def discover(self, registry: MarketRegistry) -> None:
        try:
            raw_markets = await clob_retry(
                lambda: fetch_crypto_markets(self.config.symbols),
                operation="discover_markets")
        except Exception as e:
            logger.error("discovery_failed", error=str(e))
            return

        default_slot_duration = getattr(self.config, "slot_duration", 15 * 60)
        new_cids: list[str] = []
        for m in raw_markets:
            if registry.get(m.condition_id):
                continue
            chainlink_symbol, slot_ts, slot_duration = parse_slug_info(
                m.slug, default_slot_duration)
            ref_price = self.chainlink.get_price(chainlink_symbol) or 0.0
            state = MarketState(
                condition_id=m.condition_id,
                slug=m.slug,
                symbol=chainlink_symbol,
                slot_ts=slot_ts,
                token_ids=getattr(m, "token_ids", {}),
                ref_price=ref_price,
                chainlink_symbol=chainlink_symbol,
                slot_duration=slot_duration,
            )
            registry.register(state)
            new_cids.append(m.condition_id)

        if new_cids:
            await self.poly_feed.subscribe_batch(new_cids)
            logger.info("markets_discovered", count=len(new_cids))
