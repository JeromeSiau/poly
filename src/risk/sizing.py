"""Position sizing utilities using fractional Kelly criterion.

Provides standalone sizing functions that can be used by any strategy engine
without coupling to a specific risk manager class.
"""

import structlog

logger = structlog.get_logger()


def kelly_position_size(
    capital: float,
    edge: float,
    max_pct: float = 0.10,
    liquidity: float = float("inf"),
) -> float:
    """Calculate position size using 1/4 fractional Kelly criterion.

    Parameters
    ----------
    capital:
        Available capital for the strategy.
    edge:
        Expected edge as a decimal (e.g. 0.05 for 5%).
    max_pct:
        Maximum position as a fraction of capital (default 10%).
    liquidity:
        Available market liquidity.  Position is capped at 50% of
        liquidity to limit market impact.

    Returns
    -------
    float
        Optimal position size in dollars (>= 0).
    """
    if edge <= 0 or capital <= 0:
        return 0.0

    # Kelly criterion: f* = edge / variance
    # For binary outcomes with ~50% probability, variance ~ 0.25
    kelly = (edge / 0.25) * 0.25  # 1/4 Kelly
    size = capital * kelly

    # Enforce caps
    size = min(size, capital * max_pct, liquidity * 0.5)
    size = max(0.0, size)

    logger.debug(
        "kelly_position_size",
        edge=edge,
        kelly_fraction=kelly,
        raw_size=capital * kelly,
        capped_size=size,
    )

    return size
