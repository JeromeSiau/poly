"""Generic trade persistence for all strategies.

Extracted from TwoSidedPaperRecorder. Writes to LiveObservation + PaperTrade
tables with the same game_state JSON format so the /trades API and dashboards
keep working unchanged.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

import structlog

from src.db.database import get_sync_session, init_db
from src.exceptions import PersistenceError
from src.db.models import LiveObservation, PaperTrade
from src.execution.models import TradeIntent, FillResult
from src.utils.parsing import _ensure_sync_db_url as _ensure_sync_url

logger = structlog.get_logger()


class TradeRecorder:
    """Generic trade persistence to LiveObservation + PaperTrade."""

    def __init__(
        self,
        db_url: str,
        *,
        strategy_tag: str,
        event_type: str = "",
        run_id: str = "",
        paper: bool = True,
    ) -> None:
        self._db_url = _ensure_sync_url(db_url)
        self._strategy_tag = strategy_tag
        self._event_type = event_type
        self._run_id = run_id
        self._paper = paper

    def bootstrap(self) -> None:
        """Create tables and run migrations."""
        init_db(self._db_url)

    def record_fill(
        self,
        *,
        intent: TradeIntent,
        fill: FillResult,
        fair_prices: dict[str, float] | None = None,
        execution_mode: str = "paper",
        extra_state: dict[str, Any] | None = None,
    ) -> int:
        """Persist a fill (entry). Returns observation_id, or 0 if skipped."""
        if fill.shares <= 0:
            return 0
        return self._persist(
            intent=intent,
            fill=fill,
            fair_prices=fair_prices,
            execution_mode=execution_mode,
            is_settle=False,
            extra_state=extra_state,
        )

    def record_settle(
        self,
        *,
        intent: TradeIntent,
        fill: FillResult,
        fair_prices: dict[str, float] | None = None,
        extra_state: dict[str, Any] | None = None,
    ) -> int:
        """Persist a settlement (exit). Closes related BUY records."""
        if fill.shares <= 0:
            return 0
        return self._persist(
            intent=intent,
            fill=fill,
            fair_prices=fair_prices,
            execution_mode="paper_settlement" if self._paper else "live_settlement",
            is_settle=True,
            extra_state=extra_state,
        )

    def _persist(
        self,
        *,
        intent: TradeIntent,
        fill: FillResult,
        fair_prices: dict[str, float] | None,
        execution_mode: str,
        is_settle: bool,
        extra_state: dict[str, Any] | None = None,
    ) -> int:
        fair = fair_prices or {}
        fair_price = fair.get(intent.outcome, intent.price)
        obs_ts = datetime.fromtimestamp(intent.timestamp, tz=timezone.utc)

        game_state: dict[str, Any] = {
            "strategy": self._event_type,
            "strategy_tag": self._strategy_tag,
            "run_id": self._run_id,
            "condition_id": intent.condition_id,
            "title": intent.title,
            "outcome": intent.outcome,
            "token_id": intent.token_id,
            "side": intent.side,
            "reason": intent.reason,
            "mode": execution_mode,
            "fair_price": fair_price,
            "edge_theoretical": intent.edge_pct,
            "fill_price": fill.avg_price,
            "shares": fill.shares,
            "size_usd": intent.size_usd,
        }
        if extra_state:
            game_state.update(extra_state)

        # PnL fields for sells
        edge_realized: Optional[float] = None
        pnl: Optional[float] = None
        exit_price: Optional[float] = None
        is_open = True
        closed_at: Optional[datetime] = None

        if intent.side == "SELL":
            pnl = fill.pnl_delta
            exit_price = fill.avg_price
            is_open = False
            closed_at = obs_ts
            if intent.size_usd > 0 and not is_settle:
                edge_realized = pnl / intent.size_usd

        observation = LiveObservation(
            timestamp=obs_ts,
            match_id=intent.condition_id,
            event_type=self._event_type,
            game_state=game_state,
            model_prediction=fair_price,
            polymarket_price=fill.avg_price,
        )

        trade = PaperTrade(
            observation_id=0,
            side=intent.side,
            entry_price=intent.price,
            simulated_fill_price=fill.avg_price,
            size=intent.size_usd,
            edge_theoretical=intent.edge_pct,
            edge_realized=edge_realized,
            exit_price=exit_price,
            pnl=pnl,
            is_open=is_open,
            created_at=obs_ts,
            closed_at=closed_at,
        )

        session = get_sync_session(self._db_url)
        try:
            session.add(observation)
            session.flush()
            trade.observation_id = int(observation.id)
            session.add(trade)
            if intent.side == "SELL":
                self._close_buy_records(session, intent.condition_id, closed_at)
            session.commit()
            return int(observation.id)
        except Exception as exc:
            session.rollback()
            raise PersistenceError(str(exc)) from exc
        finally:
            session.close()

    def _close_buy_records(
        self,
        session: Any,
        condition_id: str,
        closed_at: Optional[datetime],
    ) -> None:
        """Close all open BUY PaperTrade records for a given condition_id + strategy."""
        from sqlalchemy import and_, func

        strategy_filter = func.json_extract(
            LiveObservation.game_state, "$.strategy_tag"
        ) == self._strategy_tag

        buy_obs_ids = (
            session.query(LiveObservation.id)
            .filter(
                LiveObservation.match_id == condition_id,
                LiveObservation.event_type == self._event_type,
                strategy_filter,
            )
            .all()
        )
        if not buy_obs_ids:
            return
        obs_id_list = [row[0] for row in buy_obs_ids]
        session.query(PaperTrade).filter(
            and_(
                PaperTrade.observation_id.in_(obs_id_list),
                PaperTrade.side == "BUY",
                PaperTrade.is_open.is_(True),
            )
        ).update(
            {"is_open": False, "closed_at": closed_at},
            synchronize_session="fetch",
        )
