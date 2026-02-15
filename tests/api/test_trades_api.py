"""Tests for trades_api: mode/is_open filters, /balance, paper /winrate."""

from __future__ import annotations

import tempfile
import os
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient

from src.db.models import Base, LiveObservation as LO, PaperTrade as PT


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def db_url():
    """Create a tmp SQLite DB with three test records and return its URL."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    url = f"sqlite:///{tmp.name}"

    engine = create_engine(url)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    now = datetime.now(timezone.utc)

    # 1) Live closed trade  (autopilot, pnl=20, is_open=False)
    obs_live = LO(
        match_id="m1",
        event_type="crypto_td",
        game_state={"mode": "autopilot", "strategy_tag": "td_v1", "title": "BTC Up"},
        model_prediction=0.85,
        polymarket_price=0.80,
        timestamp=now - timedelta(hours=1),
    )
    session.add(obs_live)
    session.flush()

    trade_live = PT(
        observation_id=obs_live.id,
        side="BUY",
        entry_price=0.80,
        simulated_fill_price=0.80,
        size=100.0,
        edge_theoretical=0.05,
        pnl=20.0,
        is_open=False,
        closed_at=now - timedelta(minutes=30),
    )
    session.add(trade_live)

    # 2) Paper open trade  (paper, pnl=None, is_open=True)
    obs_paper_open = LO(
        match_id="m2",
        event_type="fear_selling",
        game_state={"mode": "paper", "strategy_tag": "fear_v1", "title": "ETH Down"},
        model_prediction=0.60,
        polymarket_price=0.55,
        timestamp=now - timedelta(hours=2),
    )
    session.add(obs_paper_open)
    session.flush()

    trade_paper_open = PT(
        observation_id=obs_paper_open.id,
        side="BUY",
        entry_price=0.55,
        simulated_fill_price=0.55,
        size=50.0,
        edge_theoretical=0.05,
        pnl=None,
        is_open=True,
    )
    session.add(trade_paper_open)

    # 3) Paper closed trade  (paper, pnl=-5, is_open=False)
    obs_paper_closed = LO(
        match_id="m3",
        event_type="fear_selling",
        game_state={"mode": "paper", "strategy_tag": "fear_v1", "title": "SOL Up"},
        model_prediction=0.50,
        polymarket_price=0.45,
        timestamp=now - timedelta(hours=3),
    )
    session.add(obs_paper_closed)
    session.flush()

    trade_paper_closed = PT(
        observation_id=obs_paper_closed.id,
        side="BUY",
        entry_price=0.45,
        simulated_fill_price=0.45,
        size=50.0,
        edge_theoretical=0.05,
        pnl=-5.0,
        is_open=False,
        closed_at=now - timedelta(hours=1),
    )
    session.add(trade_paper_closed)

    session.commit()
    session.close()
    engine.dispose()

    yield url

    os.unlink(tmp.name)


@pytest.fixture()
def client(db_url):
    """TestClient that points the API at the temp DB."""
    import src.api.trades_api as api_mod
    import src.db.database as db_mod
    from config.settings import settings

    # Reset global engine/session singletons so init_db picks up our URL
    db_mod.reset_engines()

    original_db_url = settings.DATABASE_URL
    settings.DATABASE_URL = db_url

    with TestClient(api_mod.app) as c:
        yield c

    # Restore
    settings.DATABASE_URL = original_db_url
    db_mod.reset_engines()


# ---------------------------------------------------------------------------
# Task 2: mode and is_open filters on /trades
# ---------------------------------------------------------------------------

class TestTradesFilters:
    def test_no_filters_returns_all(self, client):
        resp = client.get("/trades", params={"hours": 24})
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 3

    def test_mode_live(self, client):
        resp = client.get("/trades", params={"hours": 24, "mode": "live"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 1
        assert data["trades"][0]["strategy_tag"] == "td_v1"
        assert data["filters"]["mode"] == "live"

    def test_mode_paper(self, client):
        resp = client.get("/trades", params={"hours": 24, "mode": "paper"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 2
        tags = {t["strategy_tag"] for t in data["trades"]}
        assert tags == {"fear_v1"}
        assert data["filters"]["mode"] == "paper"

    def test_is_open_true(self, client):
        resp = client.get("/trades", params={"hours": 24, "is_open": "true"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 1
        assert data["trades"][0]["is_open"] is True
        assert data["filters"]["is_open"] is True

    def test_is_open_false(self, client):
        resp = client.get("/trades", params={"hours": 24, "is_open": "false"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 2
        for t in data["trades"]:
            assert t["is_open"] is False
        assert data["filters"]["is_open"] is False

    def test_mode_and_is_open_combined(self, client):
        resp = client.get("/trades", params={"hours": 24, "mode": "paper", "is_open": "false"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 1
        assert data["trades"][0]["pnl"] == -5.0


# ---------------------------------------------------------------------------
# Task 3: /balance endpoint
# ---------------------------------------------------------------------------

class TestBalance:
    def test_balance_paper(self, client):
        """Paper balance = PAPER_STARTING_CAPITAL + sum(closed pnl)."""
        resp = client.get("/balance", params={"mode": "paper"})
        assert resp.status_code == 200
        data = resp.json()
        # Default PAPER_STARTING_CAPITAL=1000, closed paper pnl = -5.0
        # Live closed trade (pnl=20) should NOT count because it's live mode
        # Only paper closed trade pnl=-5 counts
        assert data["mode"] == "paper"
        assert data["balance"] == pytest.approx(995.0, abs=0.01)

    def test_balance_default_is_paper(self, client):
        """Default mode should be paper."""
        resp = client.get("/balance")
        assert resp.status_code == 200
        data = resp.json()
        assert data["mode"] == "paper"


# ---------------------------------------------------------------------------
# Task 4: paper winrate on /winrate
# ---------------------------------------------------------------------------

class TestWinratePaper:
    def test_winrate_paper(self, client):
        resp = client.get("/winrate", params={"hours": 24, "mode": "paper"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["mode"] == "paper"
        # Only 1 paper closed trade (pnl=-5) => 0 wins, 1 loss
        assert data["losses"] == 1
        assert data["wins"] == 0
        assert data["still_open"] == 1
        assert data["winrate"] == 0.0
        assert data["total_pnl"] == pytest.approx(-5.0, abs=0.01)
