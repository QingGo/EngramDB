#!/bin/bash
# A/B sweep for the EngramDB store backend in SGLang main.
#
#   staging=bf16 : the path Session 45 measured (host fp8->bf16, 320 B/row over PCIe)
#   staging=fp8  : the optimized path (pinned fp8 slab, one H2D that casts on device)
#
# One process per cell: the store is reopened per run and the page cache is shared.
set -u
cd "$(dirname "$0")"
PY=${PY:-/root/engram-serve/venv-sg/bin/python}
TOKENS=${TOKENS:-"1 8 32 128"}
STAGINGS=${STAGINGS:-"bf16 fp8"}
ITERS=${ITERS:-25}

for S in $STAGINGS; do
  for T in $TOKENS; do
    echo "===== staging=$S tokens=$T ====="
    timeout 1800 "$PY" sc_main_engramdb_probe.py --sections geometry,graph \
      --tokens "$T" --iters "$ITERS" --correct-iters 5 --staging "$S" \
      --out "eg_${S}_${T}.json" 2>&1 |
      grep -E "^SUMMARY graph|^!!!|Error" | cut -c1-260
  done
done
echo "SWEEP DONE"
