#!/usr/bin/env bash
# Crypto Two-Sided Arbitrage launcher.
#
# Usage:
#   bin/run_crypto_two_sided.sh [budget] [min_edge] [tag] [paper|live]
#
# Examples:
#   bin/run_crypto_two_sided.sh                          # defaults
#   bin/run_crypto_two_sided.sh 100 0.005 c2s_test paper
#   bin/run_crypto_two_sided.sh 500 0.01 c2s_prod live

set -euo pipefail
cd "$(dirname "$0")/.."

BUDGET="${1:-200}"
MIN_EDGE="${2:-0.01}"
TAG="${3:-crypto_2s}"
MODE="${4:-paper}"

EXTRA_FLAGS="--paper"
if [ "$MODE" = "live" ]; then
    EXTRA_FLAGS="--autopilot"
fi

exec ./run scripts/run_crypto_two_sided.py \
    --budget "$BUDGET" \
    --min-edge "$MIN_EDGE" \
    --tag "$TAG" \
    $EXTRA_FLAGS
