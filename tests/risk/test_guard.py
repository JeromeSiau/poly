"""Tests for RiskGuard circuit breaker and daily halt."""

import time
from unittest.mock import AsyncMock

import pytest

from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

from src.db.models import Base, RiskState
from src.risk.guard import RiskGuard


@pytest.fixture
def db_url(tmp_path):
    """Return a temporary SQLite DB URL for testing."""
    return f"sqlite+aiosqlite:///{tmp_path / 'test_risk.db'}"


async def _make_guard(
    db_url: str,
    strategy_tag: str = "test_strat",
    max_consecutive_losses: int = 5,
    max_drawdown_usd: float = -50.0,
    stale_seconds: float = 30.0,
    daily_loss_limit_usd: float = -200.0,
    telegram_alerter=None,
) -> RiskGuard:
    """Create and initialize a RiskGuard for testing."""
    guard = RiskGuard(
        strategy_tag=strategy_tag,
        db_url=db_url,
        max_consecutive_losses=max_consecutive_losses,
        max_drawdown_usd=max_drawdown_usd,
        stale_seconds=stale_seconds,
        daily_loss_limit_usd=daily_loss_limit_usd,
        telegram_alerter=telegram_alerter,
    )
    await guard.initialize()
    return guard


@pytest.mark.asyncio
async def test_record_win_resets_consecutive_losses(db_url):
    """2 losses then 1 win should reset consecutive_losses to 0."""
    guard = await _make_guard(db_url, max_consecutive_losses=5)

    await guard.record_result(pnl=-5.0, won=False)
    await guard.record_result(pnl=-5.0, won=False)
    assert guard.consecutive_losses == 2

    await guard.record_result(pnl=10.0, won=True)
    assert guard.consecutive_losses == 0


@pytest.mark.asyncio
async def test_circuit_break_on_consecutive_losses(db_url):
    """3 losses (threshold=3) should trigger circuit break with reason."""
    guard = await _make_guard(db_url, max_consecutive_losses=3)

    await guard.record_result(pnl=-5.0, won=False)
    await guard.record_result(pnl=-5.0, won=False)
    assert guard.circuit_broken is False

    allowed = await guard.record_result(pnl=-5.0, won=False)
    assert allowed is False
    assert guard.circuit_broken is True
    assert "consecutive losses" in guard.circuit_reason.lower()


@pytest.mark.asyncio
async def test_circuit_break_on_drawdown(db_url):
    """Single -60$ loss (threshold=-50) should trigger circuit break."""
    guard = await _make_guard(db_url, max_drawdown_usd=-50.0)

    allowed = await guard.record_result(pnl=-60.0, won=False)
    assert allowed is False
    assert guard.circuit_broken is True
    assert "drawdown" in guard.circuit_reason.lower()


@pytest.mark.asyncio
async def test_is_trading_allowed_when_ok(db_url):
    """Fresh guard with recent book update should allow trading."""
    guard = await _make_guard(db_url)

    allowed = await guard.is_trading_allowed(last_book_update=time.time())
    assert allowed is True


@pytest.mark.asyncio
async def test_is_trading_allowed_stale(db_url):
    """Old book update (10s old, threshold 5s) should block trading."""
    guard = await _make_guard(db_url, stale_seconds=5.0)

    allowed = await guard.is_trading_allowed(last_book_update=time.time() - 10.0)
    assert allowed is False


@pytest.mark.asyncio
async def test_is_trading_allowed_circuit_broken(db_url):
    """After circuit break, is_trading_allowed should return False."""
    guard = await _make_guard(db_url, max_consecutive_losses=1)

    await guard.record_result(pnl=-5.0, won=False)
    assert guard.circuit_broken is True

    allowed = await guard.is_trading_allowed(last_book_update=time.time())
    assert allowed is False


@pytest.mark.asyncio
async def test_global_daily_halt(db_url):
    """Two guards sharing same DB, combined losses exceed limit -> halt."""
    guard_a = await _make_guard(
        db_url,
        strategy_tag="strat_a",
        daily_loss_limit_usd=-100.0,
        max_drawdown_usd=-500.0,  # high so only global halt triggers
        max_consecutive_losses=100,
    )
    guard_b = await _make_guard(
        db_url,
        strategy_tag="strat_b",
        daily_loss_limit_usd=-100.0,
        max_drawdown_usd=-500.0,
        max_consecutive_losses=100,
    )

    # Each loses 60 -> combined -120, limit is -100
    await guard_a.record_result(pnl=-60.0, won=False)
    allowed = await guard_b.record_result(pnl=-60.0, won=False)

    assert allowed is False
    # At least one guard should be circuit broken with global halt reason
    assert guard_b.circuit_broken is True
    assert "global daily halt" in guard_b.circuit_reason.lower()


@pytest.mark.asyncio
async def test_heartbeat_updates(db_url):
    """heartbeat() should update last_heartbeat in the DB."""
    guard = await _make_guard(db_url)

    await guard.heartbeat()

    # Read the DB via SQLAlchemy to verify
    engine = create_async_engine(db_url, echo=False)
    factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as session:
        result = await session.execute(
            select(RiskState).where(RiskState.strategy_tag == guard.strategy_tag)
        )
        row = result.scalar_one_or_none()
        assert row is not None
        assert row.last_heartbeat is not None
        assert abs(row.last_heartbeat - time.time()) < 2.0
    await engine.dispose()


@pytest.mark.asyncio
async def test_daily_reset_on_new_day(db_url):
    """Backdate updated_at to yesterday, re-initialize -> daily_pnl == 0."""
    guard = await _make_guard(db_url)

    # Record some losses
    await guard.record_result(pnl=-20.0, won=False)
    assert guard.daily_pnl == -20.0

    # Backdate updated_at to yesterday via SQLAlchemy
    yesterday = time.time() - 86400 * 1.5
    engine = create_async_engine(db_url, echo=False)
    factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as session:
        result = await session.execute(
            select(RiskState).where(RiskState.strategy_tag == guard.strategy_tag)
        )
        row = result.scalar_one_or_none()
        row.updated_at = yesterday
        await session.commit()
    await engine.dispose()

    # Re-initialize (simulates new process start on a new day)
    guard2 = await _make_guard(db_url)
    assert guard2.daily_pnl == 0.0


@pytest.mark.asyncio
async def test_staleness_is_temporary(db_url):
    """Stale -> False, but circuit_broken stays False. Fresh update -> True again."""
    guard = await _make_guard(db_url, stale_seconds=5.0)

    # Stale book update
    stale_time = time.time() - 10.0
    allowed = await guard.is_trading_allowed(last_book_update=stale_time)
    assert allowed is False
    assert guard.circuit_broken is False  # staleness is NOT a circuit break

    # Fresh book update -> allowed again
    allowed = await guard.is_trading_allowed(last_book_update=time.time())
    assert allowed is True


@pytest.mark.asyncio
async def test_telegram_alert_sent_on_circuit_break(db_url):
    """Telegram alert should be sent once when circuit breaks."""
    alerter = AsyncMock()
    alerter.send_custom_alert = AsyncMock(return_value=True)

    guard = await _make_guard(
        db_url,
        max_consecutive_losses=1,
        telegram_alerter=alerter,
    )

    await guard.record_result(pnl=-5.0, won=False)
    assert guard.circuit_broken is True
    alerter.send_custom_alert.assert_called_once()

    # Verify message content
    msg = alerter.send_custom_alert.call_args[0][0]
    assert "CIRCUIT BREAKER" in msg
    assert "test_strat" in msg


@pytest.mark.asyncio
async def test_telegram_alert_sent_only_once(db_url):
    """Telegram alert should only be sent once per circuit break, not repeatedly."""
    alerter = AsyncMock()
    alerter.send_custom_alert = AsyncMock(return_value=True)

    guard = await _make_guard(
        db_url,
        max_consecutive_losses=1,
        max_drawdown_usd=-5.0,
        telegram_alerter=alerter,
    )

    # First loss triggers circuit break
    await guard.record_result(pnl=-10.0, won=False)
    assert alerter.send_custom_alert.call_count == 1
