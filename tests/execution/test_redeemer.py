# tests/execution/test_redeemer.py
"""Tests for PolymarketRedeemer – focuses on the fetch/filter logic."""

from unittest.mock import patch, MagicMock

import pytest

from src.execution.redeemer import PolymarketRedeemer, POSITIONS_API


# -- Fixtures: realistic API responses ----------------------------------------

WINNING_POSITION = {
    "conditionId": "0xaaa",
    "size": 13.32,
    "avgPrice": 0.75,
    "currentValue": 13.32,
    "percentPnl": 33.3333,
    "curPrice": 1,
    "redeemable": True,
    "negativeRisk": False,
    "outcomeIndex": 1,
    "slug": "btc-updown-win",
}

LOSING_POSITION = {
    "conditionId": "0xbbb",
    "size": 6.66,
    "avgPrice": 0.75,
    "currentValue": 0,
    "percentPnl": -100,
    "curPrice": 0,
    "redeemable": True,
    "negativeRisk": False,
    "outcomeIndex": 0,
    "slug": "eth-updown-loss",
}

NULL_PNL_POSITION = {
    "conditionId": "0xccc",
    "size": 5.0,
    "avgPrice": 0.50,
    "currentValue": 5.0,
    "percentPnl": None,
    "curPrice": 1,
    "redeemable": True,
    "negativeRisk": False,
    "outcomeIndex": 0,
    "slug": "some-market-null-pnl",
}


def _mock_response(json_data, status=200):
    resp = MagicMock()
    resp.json.return_value = json_data
    resp.status_code = status
    resp.raise_for_status.side_effect = (
        None if status < 400 else Exception(f"HTTP {status}")
    )
    return resp


# -- Tests for _fetch_redeemable (our fix) ------------------------------------

@patch("src.execution.redeemer.requests.get")
def test_fetch_redeemable_returns_all_positions(mock_get):
    """Our override must NOT filter on percentPnl."""
    positions = [WINNING_POSITION, LOSING_POSITION, NULL_PNL_POSITION]
    mock_get.return_value = _mock_response(positions)

    result = PolymarketRedeemer._fetch_redeemable("0xWALLET")
    assert len(result) == 3
    mock_get.assert_called_once()


@patch("src.execution.redeemer.requests.get")
def test_fetch_redeemable_empty(mock_get):
    """Empty API response → empty list."""
    mock_get.return_value = _mock_response([])

    result = PolymarketRedeemer._fetch_redeemable("0xWALLET")
    assert result == []


@patch("src.execution.redeemer.requests.get")
def test_fetch_redeemable_api_error_raises(mock_get):
    """API errors should propagate, not be swallowed."""
    mock_get.return_value = _mock_response([], status=500)

    with pytest.raises(Exception):
        PolymarketRedeemer._fetch_redeemable("0xWALLET")


# -- Tests for _redeem_all_sync (integration with mock service) ----------------

class FakeService:
    """Minimal mock of PolyWeb3Service for testing redeem flow."""

    def __init__(self, redeem_results=None):
        self._redeem_results = redeem_results or []
        self.positions_received = None

    def _resolve_user_address(self):
        return "0xTEST"

    def _redeem_from_positions(self, positions, batch_size):
        self.positions_received = positions
        return self._redeem_results


@patch("src.execution.redeemer.requests.get")
def test_redeem_all_sync_passes_all_positions_to_service(mock_get):
    """Verify _redeem_all_sync passes unfiltered positions to the service."""
    positions = [WINNING_POSITION, LOSING_POSITION]
    mock_get.return_value = _mock_response(positions)

    service = FakeService(redeem_results=[{"tx": "0x123"}])
    redeemer = PolymarketRedeemer(service)
    data = redeemer._redeem_all_sync(batch_size=10)

    # Service received both positions (not just winning)
    assert service.positions_received == positions
    assert len(data["results"]) == 1
    assert data["results"][0] == {"tx": "0x123"}
    assert data["positions"] == positions


@patch("src.execution.redeemer.requests.get")
def test_redeem_all_sync_handles_none_results(mock_get):
    """None results from service are mapped to failed status."""
    mock_get.return_value = _mock_response([WINNING_POSITION])

    service = FakeService(redeem_results=[None])
    redeemer = PolymarketRedeemer(service)
    data = redeemer._redeem_all_sync(batch_size=10)

    assert data["results"] == [{"status": "failed"}]
    assert data["positions"] == [WINNING_POSITION]


@patch("src.execution.redeemer.requests.get")
def test_redeem_all_sync_no_positions(mock_get):
    """No redeemable positions → empty dict."""
    mock_get.return_value = _mock_response([])

    service = FakeService()
    redeemer = PolymarketRedeemer(service)
    data = redeemer._redeem_all_sync(batch_size=10)

    assert data["results"] == []
    assert data["positions"] == []
    assert service.positions_received is None  # service not called


# -- Demonstrate the poly_web3 bug we're fixing --------------------------------

def test_poly_web3_filter_bug_losing_positions():
    """Show that poly_web3's filter drops losing redeemable positions."""
    positions = [WINNING_POSITION, LOSING_POSITION]
    # This is what poly_web3 does internally:
    filtered = [i for i in positions if i.get("percentPnl") > 0]
    assert len(filtered) == 1  # Losing position dropped!
    assert filtered[0]["slug"] == "btc-updown-win"


def test_poly_web3_filter_bug_none_pnl():
    """Show that poly_web3's filter crashes on None percentPnl."""
    positions = [NULL_PNL_POSITION]
    with pytest.raises(TypeError):
        [i for i in positions if i.get("percentPnl") > 0]
