# src/arb/__init__.py
"""Arbitrage Engines - Core trading logic."""

from .reality_arb import RealityArbEngine, ArbOpportunity
from .polymarket_executor import PolymarketExecutor
from .cross_market_arb import CrossMarketArbEngine, CrossMarketOpportunity, EvaluationResult
from .crypto_arb import CryptoArbEngine, CryptoArbOpportunity
from .no_bet_scanner import NoBetScanner, NoBetOpportunity
from .dependency_detector import DependencyDetector, MarketDependency
from .two_sided_inventory import (
    TwoSidedInventoryEngine,
    MarketSnapshot,
    OutcomeQuote,
    TradeIntent,
    InventoryState,
    FillResult,
)
from .fear_classifier import FearClassifier, ClassifiedMarket
from .fear_scanner import FearMarketScanner, FearMarketCandidate
from .fear_spike_detector import FearSpikeDetector, FearSpike
from .fear_engine import FearSellingEngine, FearTradeSignal

__all__ = [
    "RealityArbEngine",
    "ArbOpportunity",
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
    "TwoSidedInventoryEngine",
    "MarketSnapshot",
    "OutcomeQuote",
    "TradeIntent",
    "InventoryState",
    "FillResult",
    "FearClassifier",
    "ClassifiedMarket",
    "FearMarketScanner",
    "FearMarketCandidate",
    "FearSpikeDetector",
    "FearSpike",
    "FearSellingEngine",
    "FearTradeSignal",
]
