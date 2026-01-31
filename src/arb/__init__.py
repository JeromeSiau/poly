# src/arb/__init__.py
"""Arbitrage Engines - Core trading logic."""

from .reality_arb import RealityArbEngine, ArbOpportunity
from .cross_market_arb import CrossMarketArbEngine, CrossMarketOpportunity, EvaluationResult

__all__ = [
    "RealityArbEngine",
    "ArbOpportunity",
    "CrossMarketArbEngine",
    "CrossMarketOpportunity",
    "EvaluationResult",
]
