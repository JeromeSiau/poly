#!/bin/bash
set -euo pipefail

BASE="$(cd "$(dirname "$0")/.." && pwd)"
PORT="${PORT:-8501}"

cd "$BASE"
export PYTHONPATH="$BASE"
export PYTHONUNBUFFERED=1

exec "$BASE/.venv/bin/streamlit" run "$BASE/src/dashboard/app.py" \
  --server.port "$PORT" \
  --server.address 0.0.0.0 \
  --server.headless true
