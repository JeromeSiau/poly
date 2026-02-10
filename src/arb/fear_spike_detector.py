"""FearSpikeDetector — Real-time fear spike detection for NO-price drops.

Extends the SpikeDetector concept to detect sudden NO-price drops (= fear
spikes) as optimal entry points for contrarian NO bets.  Also detects
correlated cross-market spikes within the same geopolitical cluster.

A *fear spike* occurs when retail panic drives YES prices up (and thus NO
prices down) faster than fundamentals justify.  These transient drops
create buying opportunities for the NO side.

Recovery detection tracks when the NO price bounces back from a spike
low, signalling that the fear-driven move has started to reverse.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass

import structlog

logger = structlog.get_logger()


@dataclass(slots=True)
class FearSpike:
    """A detected fear spike or recovery event."""

    condition_id: str
    no_price_before: float
    no_price_now: float
    drop_pct: float
    direction: str  # "fear_spike" | "recovery"
    cluster: str
    timestamp: float


class FearSpikeDetector:
    """Monitors NO-price observations and emits signals on fear spikes.

    A fear spike is detected when the NO price drops by at least
    ``spike_threshold_pct`` relative to the oldest observation within the
    rolling window.

    After a spike fires, a cooldown period suppresses duplicate signals
    for the same condition.

    Recovery is detected when the NO price rises by at least
    ``recovery_threshold_pct`` from the lowest NO price recorded after
    a spike.

    Cluster tracking allows callers to query whether multiple markets in
    the same geopolitical cluster have spiked within a short window
    (correlated fear).
    """

    def __init__(
        self,
        spike_threshold_pct: float = 0.05,
        spike_window_seconds: float = 600.0,
        cooldown_seconds: float = 120.0,
        recovery_threshold_pct: float = 0.05,
    ) -> None:
        self._spike_threshold = max(0.001, spike_threshold_pct)
        self._window = max(1.0, spike_window_seconds)
        self._cooldown = max(0.0, cooldown_seconds)
        self._recovery_threshold = max(0.001, recovery_threshold_pct)

        # condition_id -> deque of (timestamp, no_price)
        self._history: dict[str, deque[tuple[float, float]]] = {}
        # condition_id -> timestamp of last spike signal
        self._last_spike: dict[str, float] = {}
        # condition_id -> cluster name
        self._market_clusters: dict[str, str] = {}
        # cluster -> list of FearSpike events
        self._cluster_spikes: dict[str, deque[FearSpike]] = defaultdict(
            lambda: deque(maxlen=100)
        )
        # condition_id -> lowest NO price recorded after a spike
        self._spike_low: dict[str, float] = {}
        # Latest observation timestamp (used as reference in correlated queries)
        self._latest_timestamp: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def observe(
        self,
        condition_id: str,
        no_price: float,
        timestamp: float,
        cluster: str = "other",
    ) -> list[FearSpike]:
        """Record a NO-price observation and return any detected spikes.

        Parameters
        ----------
        condition_id:
            Unique identifier for the market condition.
        no_price:
            Current NO token price (0..1).
        timestamp:
            Observation timestamp (epoch seconds).
        cluster:
            Geopolitical cluster this market belongs to.

        Returns
        -------
        list[FearSpike]
            Zero or more spike / recovery signals.
        """
        self._market_clusters[condition_id] = cluster
        self._latest_timestamp = max(self._latest_timestamp, timestamp)

        # Initialise history buffer on first observation
        if condition_id not in self._history:
            self._history[condition_id] = deque(maxlen=500)

        buf = self._history[condition_id]
        buf.append((timestamp, no_price))

        # Prune entries older than the rolling window
        cutoff = timestamp - self._window
        while buf and buf[0][0] < cutoff:
            buf.popleft()

        signals: list[FearSpike] = []

        # Track spike low for recovery detection
        if condition_id in self._spike_low:
            if no_price < self._spike_low[condition_id]:
                self._spike_low[condition_id] = no_price

        # Need at least two observations to compare
        if len(buf) < 2:
            return signals

        oldest_price = buf[0][1]

        # --- Recovery detection -------------------------------------------
        if condition_id in self._spike_low:
            spike_low = self._spike_low[condition_id]
            recovery_rise = no_price - spike_low
            if recovery_rise >= self._recovery_threshold:
                recovery = FearSpike(
                    condition_id=condition_id,
                    no_price_before=spike_low,
                    no_price_now=no_price,
                    drop_pct=recovery_rise,
                    direction="recovery",
                    cluster=cluster,
                    timestamp=timestamp,
                )
                signals.append(recovery)
                logger.info(
                    "fear_spike_recovery",
                    condition_id=condition_id,
                    spike_low=spike_low,
                    no_price_now=no_price,
                    rise=recovery_rise,
                    cluster=cluster,
                )
                # Clear the spike low so we don't fire repeatedly
                del self._spike_low[condition_id]

        # --- Fear spike detection -----------------------------------------
        drop = no_price - oldest_price  # negative when price drops

        if drop >= -self._spike_threshold:
            return signals

        # Cooldown check
        last_spike_ts = self._last_spike.get(condition_id, 0.0)
        if self._cooldown > 0 and (timestamp - last_spike_ts) < self._cooldown:
            return signals

        self._last_spike[condition_id] = timestamp

        # Track the spike low for future recovery detection
        self._spike_low[condition_id] = no_price

        spike = FearSpike(
            condition_id=condition_id,
            no_price_before=oldest_price,
            no_price_now=no_price,
            drop_pct=drop,
            direction="fear_spike",
            cluster=cluster,
            timestamp=timestamp,
        )
        signals.append(spike)
        self._cluster_spikes[cluster].append(spike)

        logger.info(
            "fear_spike_detected",
            condition_id=condition_id,
            no_price_before=oldest_price,
            no_price_now=no_price,
            drop_pct=drop,
            cluster=cluster,
        )

        return signals

    def get_correlated_spikes(
        self,
        cluster: str,
        window_seconds: float = 60.0,
    ) -> list[FearSpike]:
        """Return recent spikes in *cluster* within *window_seconds* of now.

        Parameters
        ----------
        cluster:
            Geopolitical cluster to query.
        window_seconds:
            Only include spikes whose timestamp is within this many
            seconds of the current wall-clock time.

        Returns
        -------
        list[FearSpike]
            Spikes in the cluster that occurred recently.
        """
        now = self._latest_timestamp if self._latest_timestamp > 0 else time.time()
        cutoff = now - window_seconds
        return [
            s
            for s in self._cluster_spikes.get(cluster, [])
            if s.timestamp >= cutoff
        ]
