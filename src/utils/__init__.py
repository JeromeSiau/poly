"""Utility modules for the poly project.

Sub-modules:
- logging: configure_logging() for structlog setup
- team_matching: team name normalisation and fuzzy matching
- parsing: JSON/float/datetime helpers (import directly from src.utils.parsing)
- crypto_markets: Polymarket crypto market discovery (import directly from src.utils.crypto_markets)
- fair_value: empirical P(win) estimator for crypto binary markets (import directly from src.utils.fair_value)
"""

from .logging import configure_logging
from .team_matching import (
    normalize_team_name,
    get_team_variants,
    match_team_name,
    find_team_in_dataframe,
    TEAM_ALIASES,
)

__all__ = [
    "configure_logging",
    "normalize_team_name",
    "get_team_variants",
    "match_team_name",
    "find_team_in_dataframe",
    "TEAM_ALIASES",
]
