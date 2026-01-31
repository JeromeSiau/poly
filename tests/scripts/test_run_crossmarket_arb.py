"""Tests for the cross-market arbitrage bot entry point script."""

import pytest


def test_script_imports():
    """Verify the module can be imported without errors."""
    from scripts import run_crossmarket_arb

    assert hasattr(run_crossmarket_arb, "CrossMarketArbBot")
    assert hasattr(run_crossmarket_arb, "main")


def test_crossmarketarbbot_has_required_methods():
    """Verify CrossMarketArbBot has required methods."""
    from scripts.run_crossmarket_arb import CrossMarketArbBot

    # Check instance methods exist
    assert hasattr(CrossMarketArbBot, "start")
    assert hasattr(CrossMarketArbBot, "stop")
    assert hasattr(CrossMarketArbBot, "_scan_loop")
    assert hasattr(CrossMarketArbBot, "_handle_opportunity")


def test_crossmarketarbbot_init_defaults():
    """Verify CrossMarketArbBot can be initialized with defaults."""
    from scripts.run_crossmarket_arb import CrossMarketArbBot

    bot = CrossMarketArbBot()

    assert bot.autopilot is False
    assert bot.scan_interval == 5.0
    assert bot._running is False


def test_crossmarketarbbot_init_with_args():
    """Verify CrossMarketArbBot accepts custom arguments."""
    from scripts.run_crossmarket_arb import CrossMarketArbBot

    bot = CrossMarketArbBot(autopilot=True, scan_interval=10.0)

    assert bot.autopilot is True
    assert bot.scan_interval == 10.0


def test_crossmarketarbbot_has_required_components():
    """Verify CrossMarketArbBot initializes required components."""
    from scripts.run_crossmarket_arb import CrossMarketArbBot

    bot = CrossMarketArbBot()

    # Check feeds are initialized
    assert bot.azuro_feed is not None
    assert bot.overtime_feed is not None
    assert bot.polymarket_feed is not None

    # Check matcher is initialized
    assert bot.matcher is not None

    # Check risk manager is initialized
    assert bot.risk_manager is not None

    # Check engine is initialized
    assert bot.engine is not None
