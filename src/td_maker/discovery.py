# src/td_maker/discovery.py
from __future__ import annotations
import time
from typing import Any
import httpx
import structlog
from src.td_maker.state import MarketState, MarketRegistry
from src.td_maker.resilience import clob_retry
from src.utils.crypto_markets import fetch_crypto_markets, SLUG_TO_CHAINLINK
from src.utils.parsing import parse_json_list

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
        slot_duration_sec = getattr(self.config, "slot_duration", 900)
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                raw_markets = await fetch_crypto_markets(
                    client, self.config.symbols,
                    slot_duration_sec=slot_duration_sec)
        except Exception as e:
            logger.error("discovery_failed", error=str(e))
            return

        default_slot_duration = slot_duration_sec
        new_cids: list[str] = []
        for m in raw_markets:
            # m is a dict from Gamma API
            cid = m.get("conditionId", "")
            if not cid or registry.get(cid):
                continue

            # Event slug (e.g. "btc-updown-15m-1771079400")
            events = m.get("events", [])
            slug = events[0].get("slug", "") if events else m.get("slug", "")

            # Build token_ids: {outcome: token_id}
            outcomes = parse_json_list(m.get("outcomes", []))
            clob_ids = parse_json_list(m.get("clobTokenIds", []))
            token_ids = dict(zip(outcomes, clob_ids)) if outcomes and clob_ids else {}

            chainlink_symbol, slot_ts, slot_duration = parse_slug_info(
                slug, default_slot_duration)
            ref_price = self.chainlink.get_price(chainlink_symbol) or 0.0
            state = MarketState(
                condition_id=cid,
                slug=slug,
                symbol=chainlink_symbol,
                slot_ts=slot_ts,
                token_ids=token_ids,
                ref_price=ref_price,
                chainlink_symbol=chainlink_symbol,
                slot_duration=slot_duration,
            )
            registry.register(state)
            new_cids.append(cid)

        if new_cids:
            for cid in new_cids:
                market = registry.get(cid)
                if market:
                    await self.poly_feed.subscribe_market(
                        cid, token_map=market.token_ids, send=False)
            await self.poly_feed.flush_subscriptions()
            logger.info("markets_discovered", count=len(new_cids))
