import pytest
from config.settings import Settings


def test_settings_has_azuro_config():
    s = Settings()
    assert hasattr(s, "AZURO_SUBGRAPH_URL")
    assert hasattr(s, "AZURO_POLYGON_RPC")


def test_settings_has_overtime_config():
    s = Settings()
    assert hasattr(s, "OVERTIME_SUBGRAPH_URL")
    assert hasattr(s, "OVERTIME_OPTIMISM_RPC")


def test_settings_has_anthropic_config():
    s = Settings()
    assert hasattr(s, "ANTHROPIC_API_KEY")
    assert hasattr(s, "LLM_MATCH_CONFIDENCE_THRESHOLD")
    assert s.LLM_MATCH_CONFIDENCE_THRESHOLD == 0.95


def test_settings_has_crossmarket_config():
    s = Settings()
    assert hasattr(s, "CROSSMARKET_SCAN_INTERVAL_SECONDS")
    assert hasattr(s, "CROSSMARKET_MIN_EDGE_PCT")
    assert s.CROSSMARKET_MIN_EDGE_PCT == 0.02


def test_settings_has_capital_allocation():
    s = Settings()
    assert hasattr(s, "GLOBAL_CAPITAL")
    assert hasattr(s, "CAPITAL_ALLOCATION_REALITY_PCT")
    assert hasattr(s, "CAPITAL_ALLOCATION_CROSSMARKET_PCT")
    # Should sum to 100
    assert s.CAPITAL_ALLOCATION_REALITY_PCT + s.CAPITAL_ALLOCATION_CROSSMARKET_PCT == 100.0
