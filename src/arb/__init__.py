# src/arb/__init__.py
"""Arbitrage Engines - Core trading logic."""

from .polymarket_executor import PolymarketExecutor
from .cross_market_arb import CrossMarketArbEngine, CrossMarketOpportunity, EvaluationResult
from .crypto_arb import CryptoArbEngine, CryptoArbOpportunity
from .no_bet_scanner import NoBetScanner, NoBetOpportunity
from .dependency_detector import DependencyDetector, MarketDependency
from .fear_classifier import FearClassifier, ClassifiedMarket
from .fear_scanner import FearMarketScanner, FearMarketCandidate
from .fear_spike_detector import FearSpikeDetector, FearSpike
from .fear_engine import FearSellingEngine, FearTradeSignal

__all__ = [
    "PolymarketExecutor",
    "CrossMarketArbEngine",
    "CrossMarketOpportunity",
    "EvaluationResult",
    "CryptoArbEngine",
    "CryptoArbOpportunity",
    "NoBetScanner",
    "NoBetOpportunity",
    "DependencyDetector",
    "MarketDependency",
    "FearClassifier",
    "ClassifiedMarket",
    "FearMarketScanner",
    "FearMarketCandidate",
    "FearSpikeDetector",
    "FearSpike",
    "FearSellingEngine",
    "FearTradeSignal",
]
