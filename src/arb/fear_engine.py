"""FearSellingEngine — Core orchestrator for the fear-selling strategy.

Ties together scanning, spike detection, entry/exit rules, Kelly sizing,
and cluster-based risk limits to exploit retail fear-premium on Polymarket.

Core thesis: retail overprices dramatic, low-probability geopolitical events.
This engine identifies those markets and places contrarian NO bets when the
fear score is high enough and the Kelly criterion confirms positive expected
value.

Uses:
- FearMarketScanner for market discovery and scoring
- FearSpikeDetector for real-time fear spike entry signals
- Fractional Kelly criterion for position sizing
- Cluster-based exposure limits to manage correlated risk
"""

from dataclasses import dataclass
from typing import Optional

import structlog

from src.arb.fear_classifier import FearClassifier
from src.arb.fear_scanner import FearMarketCandidate, FearMarketScanner
from src.arb.fear_spike_detector import FearSpikeDetector

logger = structlog.get_logger()


@dataclass
class FearTradeSignal:
    """A concrete trade signal emitted by the FearSellingEngine."""

    condition_id: str
    token_id: str
    title: str
    side: str  # "BUY"
    outcome: str  # "NO"
    price: float  # NO price to buy at
    size_usd: float
    edge_pct: float
    fear_score: float
    cluster: str
    trigger: str  # "scan" | "spike"


class FearSellingEngine:
    """Orchestrates fear-selling strategy: scanning, sizing, and risk limits.

    Integrates FearMarketScanner for candidate discovery, FearSpikeDetector
    for real-time spike entry signals, fractional Kelly criterion for
    position sizing, and cluster-based exposure limits for correlated risk
    management.
    """

    def __init__(
        self,
        risk_manager=None,
        executor=None,
        max_cluster_pct: float = 0.30,
        max_position_pct: float = 0.10,
        kelly_fraction: float = 0.25,
        exit_no_price: float = 0.95,
        stop_yes_price: float = 0.70,
        min_fear_score: float = 0.5,
        classifier: FearClassifier | None = None,
    ) -> None:
        self._risk_manager = risk_manager
        self._executor = executor
        self._max_cluster_pct = max_cluster_pct
        self._max_position_pct = max_position_pct
        self._kelly_fraction = kelly_fraction
        self._exit_no_price = exit_no_price
        self._stop_yes_price = stop_yes_price
        self._min_fear_score = min_fear_score

        # Internal components
        self._scanner = FearMarketScanner(
            min_fear_score=min_fear_score, classifier=classifier
        )
        self._spike_detector = FearSpikeDetector()

        # State tracking
        self._cluster_exposure: dict[str, float] = {}
        self._open_positions: dict[str, float] = {}  # condition_id -> size_usd

        logger.info(
            "fear_selling_engine_initialized",
            max_cluster_pct=max_cluster_pct,
            max_position_pct=max_position_pct,
            kelly_fraction=kelly_fraction,
            exit_no_price=exit_no_price,
            stop_yes_price=stop_yes_price,
            min_fear_score=min_fear_score,
        )

    # ------------------------------------------------------------------
    # Capital
    # ------------------------------------------------------------------

    @property
    def _available_capital(self) -> float:
        """Get available capital from risk manager, or default to 100k."""
        if self._risk_manager is not None:
            return self._risk_manager.get_available_capital("fear")
        return 100_000.0

    # ------------------------------------------------------------------
    # Kelly sizing
    # ------------------------------------------------------------------

    def compute_kelly_size(
        self,
        estimated_no_prob: float,
        no_price: float,
        available_capital: float,
    ) -> float:
        """Compute position size using fractional Kelly criterion.

        Parameters
        ----------
        estimated_no_prob:
            Estimated true probability that NO wins (0..1).
        no_price:
            Current NO token price (cost basis per share).
        available_capital:
            Capital available for sizing.

        Returns
        -------
        float
            Recommended position size in USD, or 0.0 if no edge.
        """
        p = estimated_no_prob
        q = 1.0 - p
        b = (1.0 - no_price) / no_price  # net win per dollar risked

        kelly = (p * b - q) / b
        if kelly <= 0:
            return 0.0

        fraction = kelly * self._kelly_fraction
        max_size = available_capital * self._max_position_pct
        return min(fraction * available_capital, max_size)

    # ------------------------------------------------------------------
    # Cluster limits
    # ------------------------------------------------------------------

    def check_cluster_limit(self, cluster: str, proposed_size: float) -> bool:
        """Check if adding *proposed_size* to *cluster* stays within limits.

        Parameters
        ----------
        cluster:
            Geopolitical cluster name (e.g. "iran", "russia_ukraine").
        proposed_size:
            USD size of the proposed new position.

        Returns
        -------
        bool
            True if the position is within cluster limits.
        """
        current = self._cluster_exposure.get(cluster, 0.0)
        limit = self._available_capital * self._max_cluster_pct
        return (current + proposed_size) <= limit

    # ------------------------------------------------------------------
    # Entry evaluation
    # ------------------------------------------------------------------

    def evaluate_candidate(
        self, candidate: FearMarketCandidate
    ) -> Optional[FearTradeSignal]:
        """Evaluate a scanned candidate and return a trade signal if viable.

        Checks fear score threshold, validity, Kelly sizing, and cluster
        exposure limits before emitting a signal.

        Parameters
        ----------
        candidate:
            A FearMarketCandidate from the scanner.

        Returns
        -------
        Optional[FearTradeSignal]
            A trade signal if the candidate passes all checks, else None.
        """
        if candidate.fear_score < self._min_fear_score:
            logger.debug(
                "candidate_rejected_low_fear",
                condition_id=candidate.condition_id,
                fear_score=candidate.fear_score,
                min_required=self._min_fear_score,
            )
            return None

        if not candidate.is_valid:
            logger.debug(
                "candidate_rejected_invalid",
                condition_id=candidate.condition_id,
            )
            return None

        size = self.compute_kelly_size(
            estimated_no_prob=candidate.estimated_no_probability,
            no_price=candidate.no_price,
            available_capital=self._available_capital,
        )

        if size <= 0:
            logger.debug(
                "candidate_rejected_no_edge",
                condition_id=candidate.condition_id,
            )
            return None

        if not self.check_cluster_limit(candidate.cluster, size):
            logger.debug(
                "candidate_rejected_cluster_limit",
                condition_id=candidate.condition_id,
                cluster=candidate.cluster,
            )
            return None

        signal = FearTradeSignal(
            condition_id=candidate.condition_id,
            token_id=candidate.token_id,
            title=candidate.title,
            side="BUY",
            outcome="NO",
            price=candidate.no_price,
            size_usd=size,
            edge_pct=candidate.edge_pct,
            fear_score=candidate.fear_score,
            cluster=candidate.cluster,
            trigger="scan",
        )

        logger.info(
            "fear_trade_signal_generated",
            condition_id=signal.condition_id,
            side=signal.side,
            outcome=signal.outcome,
            price=signal.price,
            size_usd=signal.size_usd,
            edge_pct=signal.edge_pct,
            fear_score=signal.fear_score,
            cluster=signal.cluster,
        )

        return signal

    # ------------------------------------------------------------------
    # Exit rules
    # ------------------------------------------------------------------

    def check_exit(
        self,
        entry_price: float,
        current_no_price: float,
        current_yes_price: float,
    ) -> tuple[bool, str]:
        """Check whether an open position should be exited.

        Parameters
        ----------
        entry_price:
            The NO price at which the position was entered.
        current_no_price:
            Current NO token price.
        current_yes_price:
            Current YES token price.

        Returns
        -------
        tuple[bool, str]
            (should_exit, reason) where reason describes the exit trigger.
        """
        if current_no_price >= self._exit_no_price:
            return (
                True,
                f"Take profit: NO at {current_no_price:.2f} >= {self._exit_no_price}",
            )

        if current_yes_price >= self._stop_yes_price:
            return (
                True,
                f"Stop loss: YES at {current_yes_price:.2f} >= {self._stop_yes_price}",
            )

        return (False, "Hold")
