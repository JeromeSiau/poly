import pytest
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.db.models import (
    Base,
    CrossMarketEvent,
    PriceSnapshot,
    CrossMarketOpportunity,
    CrossMarketTrade,
)


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


def test_cross_market_event_creation(db_session):
    event = CrossMarketEvent(
        name="Chiefs win Super Bowl LIX",
        category="sports",
        polymarket_id="0x123abc",
        azuro_condition_id="456def",
        match_confidence=0.98,
        match_method="llm",
    )
    db_session.add(event)
    db_session.commit()

    assert event.id is not None
    assert event.name == "Chiefs win Super Bowl LIX"
    assert event.match_confidence == 0.98


def test_price_snapshot_creation(db_session):
    event = CrossMarketEvent(name="Test Event", category="sports")
    db_session.add(event)
    db_session.commit()

    snapshot = PriceSnapshot(
        event_id=event.id,
        platform="polymarket",
        outcome="YES",
        price=0.45,
        liquidity=10000.0,
    )
    db_session.add(snapshot)
    db_session.commit()

    assert snapshot.id is not None
    assert snapshot.price == 0.45


def test_cross_market_opportunity_creation(db_session):
    event = CrossMarketEvent(name="Test Event", category="sports")
    db_session.add(event)
    db_session.commit()

    opp = CrossMarketOpportunity(
        event_id=event.id,
        source_platform="polymarket",
        source_price=0.42,
        source_liquidity=5000.0,
        target_platform="overtime",
        target_price=0.47,
        target_liquidity=3000.0,
        gross_edge_pct=0.048,
        fees_pct=0.01,
        gas_estimate=0.15,
        net_edge_pct=0.038,
    )
    db_session.add(opp)
    db_session.commit()

    assert opp.id is not None
    assert opp.status == "detected"
    assert opp.net_edge_pct == 0.038


def test_cross_market_trade_creation(db_session):
    event = CrossMarketEvent(name="Test Event", category="sports")
    db_session.add(event)
    db_session.commit()

    opp = CrossMarketOpportunity(
        event_id=event.id,
        source_platform="polymarket",
        source_price=0.42,
        target_platform="overtime",
        target_price=0.47,
        gross_edge_pct=0.048,
        net_edge_pct=0.038,
    )
    db_session.add(opp)
    db_session.commit()

    trade = CrossMarketTrade(
        opportunity_id=opp.id,
        source_tx_hash="0xabc123",
        source_chain="polygon",
        source_amount=850.0,
        source_price_filled=0.42,
        source_gas_paid=0.05,
        source_status="confirmed",
        target_tx_hash="0xdef456",
        target_chain="optimism",
        target_amount=850.0,
        target_price_filled=0.47,
        target_gas_paid=0.10,
        target_status="confirmed",
        execution_time_ms=1200,
        realized_edge_pct=0.035,
        realized_pnl=29.75,
    )
    db_session.add(trade)
    db_session.commit()

    assert trade.id is not None
    assert trade.realized_pnl == 29.75
