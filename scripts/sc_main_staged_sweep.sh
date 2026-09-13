#!/bin/bash
# Token sweep for the SGLang-main PLE offload probe.
#
# One process per (batch, coldness) cell: the probe rebuilds the table per arm,
# and a fresh process keeps one arm's page cache and CUDA state out of the next.
# The `direct` arm runs last and alone because it is expected to misbehave.
#
#   PYTHONPATH=<sglang-main>/python sc_main_staged_sweep.sh
set -u
cd "$(dirname "$0")"
PY=${PY:-/root/engram-serve/venv-sg/bin/python}
ARMS=${ARMS:-pinned,staged}
ITERS=${ITERS:-20}
TOKENS=${TOKENS:-"1 8 32 128"}

for T in $TOKENS; do
  for MODE in cold warm; do
    FLAG=""
    [ "$MODE" = cold ] && FLAG="--cold"
    OUT="g${T}_${MODE}.json"
    echo "===== tokens=$T $MODE ====="
    timeout 1800 "$PY" sc_main_staged_probe.py --sections graph --arms "$ARMS" \
      --iters "$ITERS" --tokens "$T" $FLAG --out "$OUT" 2>&1 |
      grep -E "^---|replay read|Error" | cut -c1-200
  done
done

if [ "${SKIP_DIRECT:-0}" != "1" ]; then
  echo "===== direct (upstream file backend on a non-HMM device) ====="
  timeout 900 "$PY" sc_main_staged_probe.py --sections "" --arms direct \
    --tokens 128 --out direct.json 2>&1 | tail -8
fi
echo "SWEEP DONE"
