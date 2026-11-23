#!/usr/bin/env bash
# Run the backtest grid with one process per target election, in parallel, then summarise.
# Each process runs 4 NUTS chains, so N elections use about 4N cores.
#
#   scripts/run_backtest_grid.sh outputs/backtest/WIP_grid "rw rw_persistent gp_current poll_average last_result" "56 28 7"
#
# Env: DRAWS, TUNE (default 500 each), ELECTIONS (default: 2015 2019 2022 2024 2025 elections).
set -euo pipefail
OUT=${1:-outputs/backtest/WIP_grid}
MODELS=${2:-"rw rw_persistent rw_nobias gp_current poll_average last_result"}
HORIZONS=${3:-"56 28 7"}
DRAWS=${DRAWS:-500}
TUNE=${TUNE:-500}
ELECTIONS=${ELECTIONS:-"2015-10-04 2019-10-06 2022-01-30 2024-03-10 2025-05-18"}
mkdir -p "$OUT/logs"
pids=()
for E in $ELECTIONS; do
  python -m src.evaluation.backtest --models $MODELS --elections "$E" --horizons $HORIZONS \
    --draws "$DRAWS" --tune "$TUNE" --out "$OUT" > "$OUT/logs/$E.log" 2>&1 &
  pids+=($!)
done
status=0
for p in "${pids[@]}"; do wait "$p" || status=1; done
python -m src.evaluation.backtest --summarise "$OUT" | tee "$OUT/summary.txt"
exit $status
