"""Maps CEX symbols (BTCUSDT, etc.) to active Polymarket 15-min crypto markets.

Polymarket has rolling 15-minute markets like:
- "Will Bitcoin go up in the next 15 minutes?"
- "Will ETH be above $X at 14:30 UTC?"

This mapper syncs active crypto markets from Polymarket and links them
to the corresponding CEX trading pair for real-time signal routing.
"""

from typing import Any, Optional

import structlog

logger = structlog.get_logger()

# Map CEX symbols to Polymarket search keywords
SYMBOL_KEYWORDS: dict[str, list[str]] = {
    "BTCUSDT": ["bitcoin", "btc"],
    "ETHUSDT": ["ethereum", "eth"],
    "SOLUSDT": ["solana", "sol"],
    "XRPUSDT": ["xrp", "ripple"],
}


class CryptoMarketMapper:
    """Links CEX trading pairs to Polymarket 15-minute crypto markets."""

    def __init__(self) -> None:
        # condition_id -> market dict
        self._active_markets: dict[str, dict[str, Any]] = {}
        # symbol -> list of condition_ids
        self._symbol_to_markets: dict[str, list[str]] = {}

    async def sync_markets(self, polymarket_feed: Any) -> int:
        """Sync active 15-min crypto markets from Polymarket.

        Fetches markets matching crypto keywords and maps them to CEX symbols.
        Returns number of markets synced.
        """
        count = 0
        markets = await polymarket_feed.get_markets(tag="crypto", active=True)

        for market in markets:
            title = market.get("title", "").lower()
            condition_id = market.get("condition_id", "")

            for symbol, keywords in SYMBOL_KEYWORDS.items():
                if any(kw in title for kw in keywords):
                    self._active_markets[condition_id] = market
                    if symbol not in self._symbol_to_markets:
                        self._symbol_to_markets[symbol] = []
                    if condition_id not in self._symbol_to_markets[symbol]:
                        self._symbol_to_markets[symbol].append(condition_id)
                        count += 1
                    break

        logger.info("crypto_markets_synced", count=count)
        return count

    def get_active_market(self, symbol: str) -> Optional[dict[str, Any]]:
        """Get the most relevant active market for a CEX symbol.

        Returns None if no active market exists for this symbol.
        """
        market_ids = self._symbol_to_markets.get(symbol, [])
        if not market_ids:
            return None
        return self._active_markets.get(market_ids[0])

    def get_token_for_direction(
        self, market: dict[str, Any], direction: str
    ) -> Optional[tuple[str, str]]:
        """Get the token_id and outcome for a given price direction.

        Args:
            market: Market dict with "tokens" list
            direction: "UP" or "DOWN"

        Returns:
            (token_id, outcome) tuple or None if direction is NEUTRAL
        """
        if direction == "NEUTRAL":
            return None

        tokens = market.get("tokens", [])
        if not tokens:
            return None

        # UP → buy YES, DOWN → buy NO
        target_outcome = "Yes" if direction == "UP" else "No"
        for token in tokens:
            if token.get("outcome") == target_outcome:
                return token["token_id"], token["outcome"]

        return None
