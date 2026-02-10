"""Tests for FearSpikeDetector — real-time fear spike detection."""

import pytest

from src.arb.fear_spike_detector import FearSpike, FearSpikeDetector


class TestFearSpikeDetector:
    def setup_method(self):
        self.detector = FearSpikeDetector(
            spike_threshold_pct=0.05,
            spike_window_seconds=600,
            cooldown_seconds=120,
        )

    def test_no_spike_on_stable_prices(self):
        """Stable NO price should not trigger a spike."""
        for i in range(10):
            spikes = self.detector.observe(
                "market1", no_price=0.70, timestamp=1000.0 + i * 10, cluster="iran"
            )
        assert spikes == []

    def test_spike_on_no_price_drop(self):
        """A sudden NO price drop (fear spike) should be detected."""
        self.detector.observe("m1", no_price=0.75, timestamp=1000.0, cluster="iran")
        spikes = self.detector.observe(
            "m1", no_price=0.65, timestamp=1005.0, cluster="iran"
        )
        assert len(spikes) == 1
        assert spikes[0].direction == "fear_spike"
        assert spikes[0].drop_pct == pytest.approx(-0.10, abs=0.01)

    def test_cooldown_prevents_duplicate(self):
        """Cooldown should prevent duplicate signals."""
        self.detector.observe("m1", no_price=0.80, timestamp=1000.0, cluster="iran")
        self.detector.observe("m1", no_price=0.70, timestamp=1005.0, cluster="iran")
        self.detector.observe("m1", no_price=0.80, timestamp=1010.0, cluster="iran")
        spikes = self.detector.observe(
            "m1", no_price=0.70, timestamp=1015.0, cluster="iran"
        )
        assert spikes == []

    def test_correlated_spike_detection(self):
        """Multiple markets in same cluster spiking = correlated spike."""
        self.detector.observe("m1", no_price=0.80, timestamp=1000.0, cluster="iran")
        self.detector.observe("m2", no_price=0.75, timestamp=1000.0, cluster="iran")
        self.detector.observe("m1", no_price=0.70, timestamp=1005.0, cluster="iran")
        self.detector.observe("m2", no_price=0.65, timestamp=1008.0, cluster="iran")
        correlated = self.detector.get_correlated_spikes("iran", window_seconds=30)
        assert len(correlated) >= 2

    def test_no_correlated_spike_across_clusters(self):
        """Spikes in different clusters should not be correlated."""
        self.detector.observe("m1", no_price=0.80, timestamp=1000.0, cluster="iran")
        self.detector.observe("m2", no_price=0.75, timestamp=1000.0, cluster="russia_ukraine")
        self.detector.observe("m1", no_price=0.70, timestamp=1005.0, cluster="iran")
        self.detector.observe("m2", no_price=0.65, timestamp=1008.0, cluster="russia_ukraine")
        iran_correlated = self.detector.get_correlated_spikes("iran", window_seconds=30)
        assert len(iran_correlated) == 1

    def test_recovery_signal(self):
        """Detect when NO price recovers after a spike."""
        self.detector.observe("m1", no_price=0.80, timestamp=1000.0, cluster="iran")
        self.detector.observe("m1", no_price=0.65, timestamp=1005.0, cluster="iran")
        recovery = self.detector.observe(
            "m1", no_price=0.78, timestamp=1200.0, cluster="iran"
        )
        assert any(s.direction == "recovery" for s in recovery)
