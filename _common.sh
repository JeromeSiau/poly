#!/bin/bash
# Shared setup for all strategy launchers.
# Source this at the top of each run_*.sh:
#   source "$(dirname "$0")/_common.sh"

BASE="$(cd "$(dirname "${BASH_SOURCE[1]}")" && pwd)"

cd "$BASE"
export PYTHONPATH="$BASE"
export PYTHONUNBUFFERED=1
mkdir -p "$BASE/logs" "$BASE/data"

# Export proxy vars from .env so httpx/websockets pick them up
if [[ -f "$BASE/.env" ]]; then
  _proxy=$(grep -m1 '^HTTPS_PROXY=' "$BASE/.env" | cut -d= -f2-)
  [[ -n "$_proxy" ]] && export HTTPS_PROXY="$_proxy" HTTP_PROXY="$_proxy"
fi

PYTHON="$BASE/.venv/bin/python"
DB_URL="sqlite+aiosqlite:///data/arb.db"

# Circuit breaker defaults (override via env vars)
CB_MAX_LOSSES="${CB_MAX_LOSSES:-5}"
CB_MAX_DRAWDOWN="${CB_MAX_DRAWDOWN:--50}"
CB_STALE_SECONDS="${CB_STALE_SECONDS:-30}"
CB_DAILY_LIMIT="${CB_DAILY_LIMIT:--200}"

# Common CB args to append to any python exec
CB_ARGS=(
  --cb-max-losses "$CB_MAX_LOSSES"
  --cb-max-drawdown "$CB_MAX_DRAWDOWN"
  --cb-stale-seconds "$CB_STALE_SECONDS"
  --cb-daily-limit "$CB_DAILY_LIMIT"
)
