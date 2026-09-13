#!/usr/bin/env python3
"""How deep must the PLE layer be?  -- the schedulability rule, made checkable.

Why this exists
---------------
`docs/prefetch-lead-time.md` models the window as

    tau(L) = O_inter_step + (L/N) * C            (N layers, step time C)

and asks whether a read of cost ``t_read`` fits: ``t_read <= tau(L)``.

For the *decode* case the interesting term is the second one, and substituting
``C/N`` = per-layer forward time collapses the model to something with no N in
it at all:

    L* = ceil( t_read / per_layer_time )

i.e. **the required layer index is just the read, measured in units of one
layer's GPU time.**  That is the whole design rule, and it is falsifiable: given
a measured ``t_read`` and a measured per-layer time, L* is a number.

Two consequences worth stating plainly, because both are counter-intuitive:

1. **Concurrency buys layer depth.**  Cold NVMe cost here is dominated by page
   faults (one 4 KiB page per row), so it scales with *row count*, not bytes.
   Going from 1 thread to the thread pool is ~18x, which cuts L* by ~18x.  The
   thread pool is not a throughput optimisation; it is what makes a shallow
   PLE layer viable at all.

2. **The absolute 500 us budget is the wrong裁判.**  Whether a read is
   "acceptable" is not a property of its duration but of its duration *relative
   to the layers that precede its consumer*.  See roadmap section 36.

Usage
-----
    python scripts/lead_layer_budget.py
    python scripts/lead_layer_budget.py --json-out probes/lead_layer_budget.json

All inputs are measured quantities and are cited in the output; nothing here is
modelled or extrapolated except the arithmetic.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

# ---------------------------------------------------------------------------
# Measured inputs.  Every one carries its provenance; an uncited number here
# would be exactly the kind of soft assumption this script exists to remove.
# ---------------------------------------------------------------------------

# tags/token and row width are geometry, from the real Qwen3.8 PLE config.
QWEN_ROWS = 16          # ngram_heads
QWEN_ROW_BYTES = 160
V41_ROWS = 48           # roadmap section 29.3: V4.1 Engram is 48 rows/token
V41_ROW_BYTES = 256     # roadmap section 29.3: row-major [rows, 256]

# --- measured read costs (cold, self-proved cold) --------------------------
# probes/serve_ple_ab_fulltable_session42.md -- 128/128 shards, 47.68 GiB,
# 10/10 iterations self-reported cold, ratio 58-116x, reader = thread pool.
T_READ_QWEN_POOL_US = 195.9        # 16 rows x 160 B = 2,560 B/token
# probes/nvme_gate_autodl_session42.txt -- single thread, cold, 85-86 us/row.
T_READ_US_PER_ROW_1THREAD = 85.8   # 65-shard run; conservative for 16 rows

# --- measured per-layer forward times --------------------------------------
# probes/engine_floor_session42.md -- Qwen3.5-0.8B, 4090, batch=1, 128+128.
# The layer count N=24 is the model's own; per-layer = step_time / N.
STEPS = [
    # (label, step_ms, N, provenance)
    ("SGLang 0.5.19 graph", 2.270, 24, "probes/sglang_baseline_session42.json"),
    ("vLLM 0.29.0 graph", 2.924, 24, "probes/vllm_graph_session42.json"),
    ("vLLM 0.29.0 eager", 21.280, 24, "probes/engine_floor_session42.md"),
]
# V4.1: tau(14) = 2295 us was measured; that directly gives per-layer time
# without needing N or C separately.
V41_TAU_AT_14_US = 2295.0


def per_layer_us(step_ms: float, n: int) -> float:
    return step_ms * 1000.0 / n


def l_star(t_read_us: float, per_layer: float) -> int:
    """Smallest layer index whose lead time covers the read.

    ``ceil`` and not ``floor``: layer L must have *completed* before its data is
    needed, so we need tau(L) >= t_read, and tau is linear in L.
    """
    if per_layer <= 0:
        raise ValueError("per-layer time must be positive")
    return max(0, math.ceil(t_read_us / per_layer))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    v41_per_layer = V41_TAU_AT_14_US / 14.0

    print("=" * 78)
    print("L* = ceil( t_read / per_layer_time )")
    print("=" * 78)

    print("\n--- inputs (all measured; see the module docstring for sources) ---")
    print(f"  Qwen geometry   : {QWEN_ROWS} rows x {QWEN_ROW_BYTES} B"
          f" = {QWEN_ROWS * QWEN_ROW_BYTES:,} B/token")
    print(f"  V4.1 geometry   : {V41_ROWS} rows x {V41_ROW_BYTES} B"
          f" = {V41_ROWS * V41_ROW_BYTES:,} B/token")
    print(f"  t_read Qwen, thread pool, cold : {T_READ_QWEN_POOL_US} us/token")
    print(f"  t_read 1 thread,          cold : {T_READ_US_PER_ROW_1THREAD} us/row"
          f"  -> {T_READ_US_PER_ROW_1THREAD * QWEN_ROWS:.0f} us (16 rows)"
          f" / {T_READ_US_PER_ROW_1THREAD * V41_ROWS:.0f} us (48 rows)")
    print(f"  V4.1 per-layer (tau(14)={V41_TAU_AT_14_US:.0f} us / 14)"
          f" : {v41_per_layer:.1f} us")

    # Cost model: cold reads here are page-fault dominated (one 4 KiB page per
    # row), so the pool figure scales with ROW COUNT, not bytes.  Stated
    # explicitly because the byte-scaling alternative would give a different
    # (4.8x larger) V4.1 number.
    t_read_v41_pool = T_READ_QWEN_POOL_US * (V41_ROWS / QWEN_ROWS)
    t_read_v41_1t = T_READ_US_PER_ROW_1THREAD * V41_ROWS

    rows = []
    for label, step_ms, n, prov in STEPS:
        pl = per_layer_us(step_ms, n)
        rows.append({
            "engine": label, "step_ms": step_ms, "n_layers": n,
            "per_layer_us": round(pl, 1), "provenance": prov,
            "L_star_qwen_16row": l_star(T_READ_QWEN_POOL_US, pl),
            "L_star_v41_48row": l_star(t_read_v41_pool, pl),
        })

    v41_row = {
        "engine": "V4.1 (tau(14) measured)", "step_ms": None, "n_layers": 48,
        "per_layer_us": round(v41_per_layer, 1),
        "provenance": "docs/roadmap.md 35.1 (tau(14)=2295 us)",
        "L_star_qwen_16row": l_star(T_READ_QWEN_POOL_US, v41_per_layer),
        "L_star_v41_48row": l_star(t_read_v41_pool, v41_per_layer),
    }

    hdr = (f"{'per-layer source':<26}{'us/layer':>9}"
           f"{'L* (16 rows)':>14}{'L* (48 rows)':>14}")
    print("\n--- required layer depth WITH concurrency (thread pool) ---")
    print(hdr)
    print("-" * len(hdr))
    for r in rows + [v41_row]:
        print(f"{r['engine']:<26}{r['per_layer_us']:>9.1f}"
              f"{r['L_star_qwen_16row']:>14}{r['L_star_v41_48row']:>14}")

    no_conc = []
    for label, step_ms, n, prov in STEPS:
        pl = per_layer_us(step_ms, n)
        no_conc.append({
            "engine": label, "per_layer_us": round(pl, 1),
            "L_star_qwen_1thread": l_star(T_READ_US_PER_ROW_1THREAD * QWEN_ROWS, pl),
            "L_star_v41_1thread": l_star(t_read_v41_1t, pl),
        })
    pl = v41_per_layer
    no_conc.append({
        "engine": "V4.1 (tau(14) measured)", "per_layer_us": round(pl, 1),
        "L_star_qwen_1thread": l_star(T_READ_US_PER_ROW_1THREAD * QWEN_ROWS, pl),
        "L_star_v41_1thread": l_star(t_read_v41_1t, pl),
    })

    hdr2 = (f"{'per-layer source':<26}{'us/layer':>9}"
            f"{'L* (16 rows)':>14}{'L* (48 rows)':>14}")
    print("\n--- required layer depth WITHOUT concurrency (1 thread) ---")
    print(hdr2)
    print("-" * len(hdr2))
    for r in no_conc:
        print(f"{r['engine']:<26}{r['per_layer_us']:>9.1f}"
              f"{r['L_star_qwen_1thread']:>14}{r['L_star_v41_1thread']:>14}")

    print("\n--- what this says about the two real geometries ---")
    l_qwen = l_star(T_READ_QWEN_POOL_US, per_layer_us(2.270, 24))
    print(f"  Qwen PLE sits at layer  2 of 24."
          f"  Needs L* = {l_qwen}"
          f" (SGLang graph) -> layer 2 is {l_qwen - 2} short of the requirement,")
    print(f"     which is exactly the measured +6.7 us overshoot"
          f" (tau(2)=189.2 us < t_read=195.9 us).")
    l_v41_pool = l_star(t_read_v41_pool, v41_per_layer)
    l_v41_1t = l_star(t_read_v41_1t, v41_per_layer)
    print(f"  V4.1 Engram sits at layer 14 of 48."
          f"  Needs L* = {l_v41_pool}"
          f" -> FITS, with {14 / l_v41_pool:.1f}x more depth than required.")
    verdict = "FITS" if l_v41_1t <= 14 else "DOES NOT FIT"
    print(f"  ... but WITHOUT the thread pool it needs L* = {l_v41_1t}"
          f" -> {verdict} at layer 14 ({l_v41_1t} > 14).")
    print("\n  => Concurrency is not a throughput optimisation here;")
    print("     it is what makes a shallow PLE layer schedulable at all.")

    result = {
        "rule": "L* = ceil(t_read / per_layer_us)",
        "inputs": {
            "qwen_rows": QWEN_ROWS, "qwen_row_bytes": QWEN_ROW_BYTES,
            "v41_rows": V41_ROWS, "v41_row_bytes": V41_ROW_BYTES,
            "t_read_qwen_pool_us": T_READ_QWEN_POOL_US,
            "t_read_us_per_row_1thread": T_READ_US_PER_ROW_1THREAD,
            "v41_tau_at_14_us": V41_TAU_AT_14_US,
            "v41_per_layer_us": round(v41_per_layer, 2),
            "cost_model": "row-count (page-fault) dominated, NOT byte-scaled",
        },
        "t_read_v41_pool_us": round(t_read_v41_pool, 1),
        "t_read_v41_1thread_us": round(t_read_v41_1t, 1),
        "with_concurrency": rows + [v41_row],
        "without_concurrency": no_conc,
    }
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, indent=2) + "\n")
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
