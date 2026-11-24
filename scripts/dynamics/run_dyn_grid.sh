#!/usr/bin/env bash
# Dynamics-only backtest grid (finish), 1000/1000 draws, into WIP_dyn_v2 (its three 2025 h7 gate
# fits are reused). Convergence is judged afterwards on the forecast draws (eval_dyn.py), not by
# stopping on the short-term scale. Markers in $OUT/run.log: LANE_DONE, STAGE2_DONE,
# STAGE3_DONE, FITS json= failed=, ALL DONE / STOPPED.
set -uo pipefail
cd "$(dirname "$0")/../.."  # repo root (script lives in scripts/dynamics)
OUT=outputs/backtest/WIP_dyn_v2
ELECTIONS="2025-05-18 2024-03-10 2022-01-30 2019-10-06 2015-10-04"
mkdir -p "$OUT/logs"
log() { echo "$(date '+%F %T') $*" >> "$OUT/run.log"; }
fit() {  # fit <models> <election> <horizons...>
  local models=$1 E=$2; shift 2
  pixi run python -m src.evaluation.backtest --models $models --elections "$E" --horizons "$@" \
    --draws 1000 --tune 1000 --out "$OUT" >> "$OUT/logs/$E.log" 2>&1
}
log "FINISH_START $(git rev-parse --short HEAD)"

# stage 2: the comparison itself, one lane per election
for E in $ELECTIONS; do ( fit "dyn_rw dyn_ou4s poll_average last_result" "$E" 7 28 56; log "LANE_DONE S2 $E" ) & done
wait
log "STAGE2_DONE"
pixi run python scripts/dynamics/eval_dyn.py "$OUT" > "$OUT/eval_stage2.txt" 2>&1 || log "EVAL_FAIL stage2"

# stage 3: neighbours (2 and 8 weeks) on every horizon, smooth Matern-3/2 at h7
for E in $ELECTIONS; do ( fit "dyn_ou2s dyn_ou8s" "$E" 7 28 56; fit "dyn_m32_4s" "$E" 7; log "LANE_DONE S3 $E" ) & done
wait
log "STAGE3_DONE"

for h in 7 28 56; do
  pixi run python -m src.evaluation.backtest --calibrate "$OUT" --models dyn_rw dyn_ou4s dyn_ou2s dyn_ou8s dyn_m32_4s --horizons $h \
    > "$OUT/calibrate_h$h.txt" 2>&1 || log "CALIBRATE_FAIL h$h"
done
pixi run python -m src.evaluation.backtest --summarise "$OUT" > "$OUT/summary.txt" 2>&1 || log "SUMMARY_FAIL"
pixi run python scripts/dynamics/eval_dyn.py "$OUT" > "$OUT/eval_final.txt" 2>&1 || log "EVAL_FAIL final"
nfail=$(grep -h "^FAILED" "$OUT"/logs/*.log | wc -l | tr -d ' ')
njson=$(ls "$OUT"/dyn_*__h0*.json 2>/dev/null | wc -l | tr -d ' ')
log "FITS json=$njson failed=$nfail"
if [ "$nfail" != "0" ] || grep -q "_FAIL" <(sed -n '/FINISH_START/,$p' "$OUT/run.log"); then log "STOPPED (failures above)"; else log "ALL DONE"; fi
