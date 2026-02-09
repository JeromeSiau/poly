import time
import pytest
from src.feeds.spike_detector import SpikeDetector, SpikeSignal


def test_detects_spike_above_threshold():
    """A 20% move in <60s triggers a spike."""
    detector = SpikeDetector(threshold_pct=0.15, window_seconds=60)
    now = time.time()

    detector.observe("cid1", "Yes", 0.50, now - 30)
    detector.observe("cid1", "Yes", 0.50, now - 20)
    signals = detector.observe("cid1", "Yes", 0.72, now)  # +22%

    assert len(signals) == 1
    assert signals[0].condition_id == "cid1"
    assert signals[0].outcome == "Yes"
    assert signals[0].direction == "up"
    assert abs(signals[0].delta - 0.22) < 0.01


def test_no_spike_below_threshold():
    """A 10% move should not trigger with 15% threshold."""
    detector = SpikeDetector(threshold_pct=0.15, window_seconds=60)
    now = time.time()

    detector.observe("cid1", "Yes", 0.50, now - 10)
    signals = detector.observe("cid1", "Yes", 0.59, now)

    assert len(signals) == 0


def test_spike_only_fires_once_per_cooldown():
    """Same condition should not re-fire within cooldown."""
    detector = SpikeDetector(threshold_pct=0.15, window_seconds=60, cooldown_seconds=120)
    now = time.time()

    detector.observe("cid1", "Yes", 0.50, now - 30)
    signals1 = detector.observe("cid1", "Yes", 0.72, now)
    assert len(signals1) == 1

    # Second spike within cooldown - should NOT fire
    detector.observe("cid1", "Yes", 0.72, now + 5)
    signals2 = detector.observe("cid1", "Yes", 0.92, now + 10)
    assert len(signals2) == 0


def test_detects_downward_spike():
    """Spike detection works for drops too (useful for the other side)."""
    detector = SpikeDetector(threshold_pct=0.15, window_seconds=60)
    now = time.time()

    detector.observe("cid1", "No", 0.50, now - 10)
    signals = detector.observe("cid1", "No", 0.28, now)

    assert len(signals) == 1
    assert signals[0].direction == "down"
    assert abs(signals[0].delta - (-0.22)) < 0.01
