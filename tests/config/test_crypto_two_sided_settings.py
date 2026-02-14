from config.settings import Settings


def test_crypto_two_sided_defaults():
    s = Settings()
    assert s.CRYPTO_TWO_SIDED_SYMBOLS == "BTCUSDT,ETHUSDT"
    assert s.CRYPTO_TWO_SIDED_TIMEFRAMES == "300,900"
    assert s.CRYPTO_TWO_SIDED_MIN_EDGE_PCT == 0.01
    assert s.CRYPTO_TWO_SIDED_BUDGET_PER_MARKET == 200.0
    assert s.CRYPTO_TWO_SIDED_MAX_CONCURRENT == 8
    assert s.CRYPTO_TWO_SIDED_ENTRY_WINDOW_S == 30
    assert s.CRYPTO_TWO_SIDED_DISCOVERY_LEAD_S == 15
    assert s.CRYPTO_TWO_SIDED_POLL_INTERVAL_S == 2.0
    assert s.CRYPTO_TWO_SIDED_FEE_BPS == 100
    assert s.CRYPTO_TWO_SIDED_PAPER_CAPITAL == 2000.0
