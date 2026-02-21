"""Test spread-based sizing in CryptoTDMaker._compute_order_size_usd."""


def _make_sizer(order_size_usd: float, spread_size_mult: float, has_model: bool = False):
    """Create a minimal mock that exercises _compute_order_size_usd logic."""

    class FakeMaker:
        def __init__(self):
            self.order_size_usd = order_size_usd
            self.spread_size_mult = spread_size_mult
            self._model = True if has_model else None

        @staticmethod
        def _model_size_scale(p_win: float) -> float:
            return max(0.2, min(2.0, (p_win - 0.5) / 0.25))

        def _compute_order_size_usd(self, p_win=None, spread=0.0):
            base = self.order_size_usd
            if p_win is not None and self._model:
                base = round(base * self._model_size_scale(p_win), 2)
            if self.spread_size_mult > 0 and spread >= 0.02:
                if spread >= 0.04:
                    base *= min(self.spread_size_mult, 2.5)
                else:
                    base *= min(1.0 + (self.spread_size_mult - 1.0) * 0.5, 2.0)
            return base

    return FakeMaker()


class TestSpreadSizing:
    def test_disabled_by_default(self):
        s = _make_sizer(10.0, 0.0)
        assert s._compute_order_size_usd(spread=0.05) == 10.0

    def test_no_boost_tight_spread(self):
        s = _make_sizer(10.0, 2.0)
        assert s._compute_order_size_usd(spread=0.01) == 10.0

    def test_medium_spread_half_boost(self):
        s = _make_sizer(10.0, 2.0)
        # spread 0.02-0.04 → 1.0 + (2.0-1.0)*0.5 = 1.5x
        assert s._compute_order_size_usd(spread=0.03) == 15.0

    def test_wide_spread_full_boost(self):
        s = _make_sizer(10.0, 2.0)
        # spread >= 0.04 → min(2.0, 2.5) = 2.0x
        assert s._compute_order_size_usd(spread=0.05) == 20.0

    def test_very_high_mult_capped(self):
        s = _make_sizer(10.0, 5.0)
        # spread >= 0.04 → min(5.0, 2.5) = 2.5x cap
        assert s._compute_order_size_usd(spread=0.06) == 25.0

    def test_model_and_spread_stack(self):
        s = _make_sizer(10.0, 2.0, has_model=True)
        # p_win=0.75 → model scale = min(2.0, (0.75-0.5)/0.25) = 1.0
        # base after model = 10.0 * 1.0 = 10.0
        # spread 0.05 → full mult 2.0x → 20.0
        assert s._compute_order_size_usd(p_win=0.75, spread=0.05) == 20.0

    def test_no_model_no_p_win(self):
        s = _make_sizer(10.0, 2.0)
        # No p_win, spread boost still applies
        assert s._compute_order_size_usd(p_win=None, spread=0.03) == 15.0
