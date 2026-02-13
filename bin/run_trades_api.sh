#!/bin/bash
set -euo pipefail

BASE="$(cd "$(dirname "$0")" && pwd)"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8788}"

cd "$BASE"
export PYTHONPATH="$BASE"

exec "$BASE/.venv/bin/uvicorn" src.api.trades_api:app --host "$HOST" --port "$PORT"
