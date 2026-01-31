import pytest
from src.risk.manager import UnifiedRiskManager


@pytest.fixture
def risk_manager():
    return UnifiedRiskManager(
        global_capital=10000.0,
        reality_allocation_pct=50.0,
        crossmarket_allocation_pct=50.0,
        max_position_pct=0.10,
        daily_loss_limit_pct=0.05,
    )


def test_get_available_capital_reality(risk_manager):
    capital = risk_manager.get_available_capital("reality")
    assert capital == 5000.0  # 50% of 10000


def test_get_available_capital_crossmarket(risk_manager):
    capital = risk_manager.get_available_capital("crossmarket")
    assert capital == 5000.0  # 50% of 10000


def test_check_position_limit_within_limit(risk_manager):
    # 10% of 5000 = 500
    assert risk_manager.check_position_limit(400.0, "reality") is True
    assert risk_manager.check_position_limit(500.0, "reality") is True


def test_check_position_limit_exceeds_limit(risk_manager):
    # 10% of 5000 = 500
    assert risk_manager.check_position_limit(600.0, "reality") is False


def test_check_daily_loss_limit_not_hit(risk_manager):
    assert risk_manager.check_daily_loss_limit() is True


def test_check_daily_loss_limit_hit(risk_manager):
    # Simulate losses
    risk_manager.record_pnl(-300.0, "reality")
    risk_manager.record_pnl(-250.0, "crossmarket")
    # Total loss = 550, limit = 5% of 10000 = 500
    assert risk_manager.check_daily_loss_limit() is False


def test_record_pnl_updates_daily_pnl(risk_manager):
    risk_manager.record_pnl(100.0, "reality")
    risk_manager.record_pnl(-50.0, "crossmarket")
    assert risk_manager.daily_pnl == 50.0


def test_calculate_position_size(risk_manager):
    size = risk_manager.calculate_position_size(
        strategy="crossmarket",
        available_liquidity=10000.0,
        edge_pct=0.05,
    )
    # Should be min(max_position, kelly_fraction * capital, liquidity * 0.5)
    assert size > 0
    assert size <= 500  # max 10% of 5000


def test_reset_daily_stats(risk_manager):
    risk_manager.record_pnl(-100.0, "reality")
    risk_manager.reset_daily_stats()
    assert risk_manager.daily_pnl == 0.0
    assert risk_manager.check_daily_loss_limit() is True
