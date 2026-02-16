from src.utils.fair_value import estimate_fair_value


def test_strong_up_move_late_slot():
    """BTC +0.2% with 3 min left -> fair value should be high (>0.70)."""
    fv = estimate_fair_value(dir_move_pct=0.2, minutes_remaining=3)
    assert fv > 0.70


def test_strong_down_move_late_slot():
    """BTC -0.3% with 2 min left -> fair value should be low (<0.30)."""
    fv = estimate_fair_value(dir_move_pct=-0.3, minutes_remaining=2)
    assert fv < 0.30


def test_flat_midslot():
    """BTC ~flat with 7 min left -> fair value near 0.50.

    Note: dir_move=0.0 falls in the [0.00, 0.05) bucket via bisect_right,
    so it reads slightly above 0.50. A tiny negative move (-0.01) falls in
    [-0.05, 0.00) and reads below 0.50. Average of both sides ~ 0.50.
    """
    fv_pos = estimate_fair_value(dir_move_pct=0.0, minutes_remaining=7)
    fv_neg = estimate_fair_value(dir_move_pct=-0.01, minutes_remaining=7)
    avg = (fv_pos + fv_neg) / 2
    assert 0.45 < avg < 0.55


def test_returns_float_in_range():
    """Fair value is always a probability in [0, 1]."""
    for move in [-1.0, -0.1, 0.0, 0.1, 1.0]:
        for mins in [1, 5, 10, 14]:
            fv = estimate_fair_value(dir_move_pct=move, minutes_remaining=mins)
            assert 0.0 <= fv <= 1.0


def test_interpolates_between_buckets():
    """Values between bucket boundaries should interpolate smoothly."""
    fv1 = estimate_fair_value(dir_move_pct=0.05, minutes_remaining=5)
    fv2 = estimate_fair_value(dir_move_pct=0.15, minutes_remaining=5)
    # More positive move -> higher fair value
    assert fv2 > fv1


def test_extreme_move_clamps_minutes():
    """Minutes outside 1-14 range should be clamped."""
    fv_low = estimate_fair_value(dir_move_pct=0.1, minutes_remaining=0)
    fv_at_1 = estimate_fair_value(dir_move_pct=0.1, minutes_remaining=1)
    assert fv_low == fv_at_1

    fv_high = estimate_fair_value(dir_move_pct=0.1, minutes_remaining=20)
    fv_at_14 = estimate_fair_value(dir_move_pct=0.1, minutes_remaining=14)
    assert fv_high == fv_at_14


def test_monotonic_with_time():
    """With a positive move, fair value should increase as time decreases (fewer min left)."""
    fvs = [estimate_fair_value(dir_move_pct=0.15, minutes_remaining=m) for m in range(14, 0, -1)]
    # Overall trend should be increasing (less time = more certainty)
    # Check first vs last
    assert fvs[-1] > fvs[0]


def test_symmetry_around_zero():
    """A +0.2% move should have roughly complementary fair value to -0.2%."""
    fv_up = estimate_fair_value(dir_move_pct=0.2, minutes_remaining=5)
    fv_down = estimate_fair_value(dir_move_pct=-0.2, minutes_remaining=5)
    # They should roughly sum to 1.0 (within tolerance for empirical data)
    assert abs((fv_up + fv_down) - 1.0) < 0.15
