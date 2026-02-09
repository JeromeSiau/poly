import pytest
from src.arb.sniper_router import SniperRouter, SniperAction
from src.feeds.spike_detector import SpikeSignal
from src.feeds.odds_api import ScoreChange
from src.analysis.event_condition_mapper import EventConditionMapper


def _build_mapper() -> EventConditionMapper:
    mapper = EventConditionMapper()
    mapper.build([
        {
            "conditionId": "cid_win",
            "question": "Will Jaguares win?",
            "events": [{"slug": "col1-jag-dep"}],
            "outcomes": '["Yes","No"]',
            "clobTokenIds": '["tok_w_y","tok_w_n"]',
        },
        {
            "conditionId": "cid_draw",
            "question": "Draw?",
            "events": [{"slug": "col1-jag-dep"}],
            "outcomes": '["Yes","No"]',
            "clobTokenIds": '["tok_d_y","tok_d_n"]',
        },
    ])
    return mapper


def test_spike_on_win_yes_triggers_siblings():
    """Spike up on Win/Yes should also trigger Draw/No buy."""
    mapper = _build_mapper()
    router = SniperRouter(mapper=mapper)

    spike = SpikeSignal(
        condition_id="cid_win", outcome="Yes",
        price_before=0.50, price_now=0.80, delta=0.30,
        direction="up", timestamp=1000.0,
    )
    actions = router.route_spike(spike)

    # Should want to buy Win/Yes AND Draw/No (if team is winning, draw less likely)
    cids = {a.condition_id for a in actions}
    assert "cid_win" in cids
    assert "cid_draw" in cids

    # Win/Yes should be BUY, Draw should be BUY No
    win_action = next(a for a in actions if a.condition_id == "cid_win")
    assert win_action.outcome == "Yes"
    assert win_action.side == "BUY"

    draw_action = next(a for a in actions if a.condition_id == "cid_draw")
    assert draw_action.outcome == "No"
    assert draw_action.side == "BUY"


def test_score_change_triggers_buy_on_all_conditions():
    """Score change should trigger trades on all event conditions."""
    mapper = _build_mapper()
    router = SniperRouter(mapper=mapper)

    change = ScoreChange(
        event_id="soccer_col1:abc",
        sport="soccer_col1",
        home_team="Jaguares",
        away_team="Deportivo Pereira",
        home_score=1, away_score=0,
        prev_home_score=0, prev_away_score=0,
        completed=False, change_type="score_change",
    )
    actions = router.route_score_change(change)

    cids = {a.condition_id for a in actions}
    assert "cid_win" in cids  # home team scored -> buy Win/Yes
    assert "cid_draw" in cids  # goal scored -> buy Draw/No
