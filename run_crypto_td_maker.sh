#!/bin/bash
set -euo pipefail

# Passive time-decay maker for Polymarket 15-min crypto markets.
#
# Watches orderbooks via WS. When bid on Up or Down enters [TARGET_BID, MAX_BID],
# places GTC buy at current bid (maker, 0 fees). Holds to resolution.
#
# Usage:
#   run_crypto_td_maker.sh                         # paper, defaults
#   run_crypto_td_maker.sh --live                  # live trading
#   run_crypto_td_maker.sh --target-bid 0.78       # custom bid level
#   run_crypto_td_maker.sh --wallet 500            # custom wallet size

BASE="$(cd "$(dirname "$0")" && pwd)"

# Defaults
SYMBOLS="BTCUSDT,ETHUSDT"
MODE="paper"
TARGET_BID="${TARGET_BID:-0.75}"
MAX_BID="${MAX_BID:-0.85}"
WALLET_USD="${WALLET_USD:-200}"

# Parse args
_positional=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --symbols)      SYMBOLS="$2"; shift 2 ;;
    --paper)        MODE="paper"; shift ;;
    --live)         MODE="live"; shift ;;
    --target-bid)   TARGET_BID="$2"; shift 2 ;;
    --max-bid)      MAX_BID="$2"; shift 2 ;;
    --wallet)       WALLET_USD="$2"; shift 2 ;;
    *)              _positional+=("$1"); shift ;;
  esac
done

TAG="crypto_td_maker_${SYMBOLS//,/_}_bid${TARGET_BID}"

cd "$BASE"
export PYTHONPATH="$BASE"
export PYTHONUNBUFFERED=1
mkdir -p "$BASE/logs" "$BASE/data"
LOG_FILE="$BASE/logs/${TAG}.log"

# Sizing: order_size = 2.5% of wallet, max_exposure = 50% of wallet
ORDER_SIZE="$(awk -v w="$WALLET_USD" 'BEGIN{v=w*0.025; if(v<1) v=1; printf "%.2f", v}')"
MAX_EXPOSURE="$(awk -v w="$WALLET_USD" 'BEGIN{v=w*0.50; if(v<50) v=50; printf "%.2f", v}')"

# Overridable intervals
DISCOVERY_INTERVAL="${DISCOVERY_INTERVAL:-60}"
MAKER_INTERVAL="${MAKER_INTERVAL:-0.5}"

MODE_FLAG="--paper"
if [[ "$MODE" == "live" ]]; then
  MODE_FLAG="--live"
fi

exec >> "$LOG_FILE" 2>&1
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] start tag=$TAG mode=$MODE symbols=$SYMBOLS target_bid=$TARGET_BID max_bid=$MAX_BID wallet=$WALLET_USD order_size=$ORDER_SIZE max_exposure=$MAX_EXPOSURE"

exec "$BASE/.venv/bin/python" "$BASE/scripts/run_crypto_td_maker.py" \
  --symbols "$SYMBOLS" \
  $MODE_FLAG \
  --target-bid "$TARGET_BID" \
  --max-bid "$MAX_BID" \
  --order-size "$ORDER_SIZE" \
  --max-exposure "$MAX_EXPOSURE" \
  --discovery-interval "$DISCOVERY_INTERVAL" \
  --maker-interval "$MAKER_INTERVAL" \
  --strategy-tag "$TAG" \
  --db-url "sqlite+aiosqlite:///data/arb.db"
