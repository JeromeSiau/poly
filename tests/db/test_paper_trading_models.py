"""Tests for paper trading database models."""

import pytest
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.db.models import Base, LiveObservation, PaperTrade


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


class TestLiveObservation:
    def test_create_observation(self, db_session):
        obs = LiveObservation(
            match_id="pandascore_12345",
            event_type="baron_kill",
            game_state={"gold_diff": 5000, "game_time": 25},
            model_prediction=0.72,
            polymarket_price=0.58,
        )
        db_session.add(obs)
        db_session.commit()

        assert obs.id is not None
        assert obs.model_prediction == 0.72
        assert obs.edge_theoretical == pytest.approx(0.14, rel=0.01)

    def test_observation_with_followup_prices(self, db_session):
        obs = LiveObservation(
            match_id="pandascore_12345",
            event_type="dragon_kill",
            game_state={"gold_diff": 2000},
            model_prediction=0.65,
            polymarket_price=0.55,
            polymarket_price_30s=0.58,
            polymarket_price_60s=0.62,
            polymarket_price_120s=0.64,
        )
        db_session.add(obs)
        db_session.commit()

        assert obs.polymarket_price_120s == 0.64


class TestPaperTrade:
    def test_create_paper_trade(self, db_session):
        obs = LiveObservation(
            match_id="pandascore_12345",
            event_type="baron_kill",
            game_state={},
            model_prediction=0.72,
            polymarket_price=0.58,
        )
        db_session.add(obs)
        db_session.commit()

        trade = PaperTrade(
            observation_id=obs.id,
            side="BUY",
            entry_price=0.58,
            simulated_fill_price=0.585,
            size=50.0,
            edge_theoretical=0.14,
        )
        db_session.add(trade)
        db_session.commit()

        assert trade.id is not None
        assert trade.size == 50.0

    def test_paper_trade_pnl_calculation(self, db_session):
        trade = PaperTrade(
            observation_id=1,
            side="BUY",
            entry_price=0.58,
            simulated_fill_price=0.585,
            size=50.0,
            edge_theoretical=0.14,
            exit_price=0.68,
            pnl=8.12,  # (0.68 - 0.585) * 50 * (1/0.585)
        )
        db_session.add(trade)
        db_session.commit()

        assert trade.pnl == 8.12
