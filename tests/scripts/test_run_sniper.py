import asyncio
import time
import pytest
from src.feeds.spike_detector import SpikeDetector, SpikeSignal
from src.feeds.odds_api import ScoreTracker, LiveGame, ScoreChange
from src.analysis.event_condition_mapper import EventConditionMapper
from src.arb.sniper_router import SniperRouter, SniperAction


@pytest.mark.asyncio
async def test_sniper_end_to_end_spike_to_actions():
    """Full pipeline: market data -> mapper -> spike -> router -> actions."""
    # 1. Build mapper from fake Gamma data
    mapper = EventConditionMapper()
    mapper.build([
        {
            "conditionId": "cid_win",
            "question": "Will Team A win?",
            "events": [{"slug": "epl-tea-teb"}],
            "outcomes": '["Yes","No"]',
            "clobTokenIds": '["tok1","tok2"]',
        },
        {
            "conditionId": "cid_draw",
            "question": "Will it end in a draw?",
            "events": [{"slug": "epl-tea-teb"}],
            "outcomes": '["Yes","No"]',
            "clobTokenIds": '["tok3","tok4"]',
        },
    ])

    # 2. Simulate spike on cid_win
    detector = SpikeDetector(threshold_pct=0.15, window_seconds=60)
    now = time.time()
    detector.observe("cid_win", "Yes", 0.45, now - 20)
    spikes = detector.observe("cid_win", "Yes", 0.78, now)
    assert len(spikes) == 1

    # 3. Route spike to actions
    router = SniperRouter(mapper=mapper)
    actions = router.route_spike(spikes[0])

    assert len(actions) >= 2
    cids = {a.condition_id for a in actions}
    assert "cid_win" in cids
    assert "cid_draw" in cids

    # 4. Check action details
    win_action = next(a for a in actions if a.condition_id == "cid_win")
    assert win_action.side == "BUY"
    assert win_action.outcome == "Yes"
    assert win_action.reason == "spike_primary"

    draw_action = next(a for a in actions if a.condition_id == "cid_draw")
    assert draw_action.side == "BUY"
    assert draw_action.outcome == "No"
    assert draw_action.reason == "spike_sibling"


@pytest.mark.asyncio
async def test_sniper_score_change_pipeline():
    """Full pipeline: score change -> router -> actions on all conditions."""
    mapper = EventConditionMapper()
    mapper.build([
        {
            "conditionId": "cid_win",
            "question": "Will Liverpool win?",
            "events": [{"slug": "epl-liv-ars"}],
            "outcomes": '["Yes","No"]',
            "clobTokenIds": '["tok1","tok2"]',
        },
        {
            "conditionId": "cid_draw",
            "question": "Will it be a draw?",
            "events": [{"slug": "epl-liv-ars"}],
            "outcomes": '["Yes","No"]',
            "clobTokenIds": '["tok3","tok4"]',
        },
    ])

    tracker = ScoreTracker()
    # First poll: set baseline
    game1 = LiveGame(
        event_id="epl:abc", sport="soccer_epl",
        home_team="Liverpool", away_team="Arsenal",
        home_score=0, away_score=0, completed=False,
    )
    changes = tracker.update([game1])
    assert len(changes) == 0

    # Second poll: Liverpool scored
    game2 = LiveGame(
        event_id="epl:abc", sport="soccer_epl",
        home_team="Liverpool", away_team="Arsenal",
        home_score=1, away_score=0, completed=False,
    )
    changes = tracker.update([game2])
    assert len(changes) == 1

    router = SniperRouter(mapper=mapper)
    actions = router.route_score_change(changes[0])
    cids = {a.condition_id for a in actions}
    assert "cid_win" in cids
    assert "cid_draw" in cids
