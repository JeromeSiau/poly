"""FastAPI endpoint exposing local-vs-RN1 behavior comparison."""

from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, HTTPException, Query

from src.analysis.rn1_comparison import DEFAULT_RN1_WALLET, build_comparison_report

app = FastAPI(title="RN1 Comparison API", version="1.0.0")


@app.get("/health")
def health() -> dict[str, bool]:
    return {"ok": True}


@app.get("/compare/rn1")
def compare_rn1(
    db: str = Query(default="data/arb.db", description="SQLite path or SQLAlchemy URL."),
    hours: float = Query(default=6.0, ge=0.1, le=168.0, description="Comparison window in hours."),
    strategy_tag: Optional[str] = Query(default=None, description="Optional local strategy tag filter."),
    rn1_wallet: str = Query(default=DEFAULT_RN1_WALLET, description="RN1 wallet (or another benchmark wallet)."),
    page_limit: int = Query(default=500, ge=50, le=500, description="Rows per RN1 activity page."),
    max_pages: int = Query(default=7, ge=1, le=20, description="Max RN1 activity pages to fetch."),
) -> dict:
    try:
        return build_comparison_report(
            db_url=db,
            window_hours=hours,
            strategy_tag=strategy_tag,
            rn1_wallet=rn1_wallet,
            page_limit=page_limit,
            max_pages=max_pages,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"comparison_failed: {exc}") from exc
