#!/usr/bin/env python3
"""Lead-time budget: can earlier layers hide the memory read?

Computes, for a memory module at layer ``L`` of an ``N``-layer model, whether
its per-step fetch cost ``F`` fits inside the window between "this token is
known" and "layer L needs the data":

    tau(L) = O + (L / N) * C          lead time
    hidden <=> F <= tau(L)            fully hidden
    exposed  = max(0, F - tau(L))     what lands on the critical path
    relative = exposed / (O + C)      as a fraction of the step

Derivation and the V4.1 numbers live in ``docs/prefetch-lead-time.md``.

Every parameter is explicit and labelled: the point of this script is that no
number in the doc is hand-copied, and that assumptions (notably ``--layers``
for V4.1) are visible at the call site.

Usage
-----
    # V4.1 Engram, two modules at layer 1 and 14, measured Store-I fetch cost
    python scripts/prefetch_lead_time.py --layers 61 --modules 1:220,14:220 \
        --tok-per-s 100

    # sweep the throughput target and show where hiding stops working
    python scripts/prefetch_lead_time.py --layers 61 --modules 1:220,14:220 \
        --sweep 100,200,400,800
"""

from __future__ import annotations

import argparse
import json

BUDGET_US = 500.0  # = 5% of 100 tok/s, the repo's standing per-token budget


def evaluate(
    *,
    layers: int,
    layer: int,
    fetch_us: float,
    step_us: float,
    overhead_us: float,
    compute_us: float,
) -> dict:
    tau = overhead_us + (layer / layers) * compute_us
    exposed = max(0.0, fetch_us - tau)
    return {
        "layer": layer,
        "fetch_us": fetch_us,
        "tau_us": tau,
        "fully_hidden": fetch_us <= tau,
        "exposed_us": exposed,
        "exposed_pct_of_step": 100.0 * exposed / step_us,
        "budget_pct": 100.0 * fetch_us / BUDGET_US,
        # IOPS a medium must sustain to hit tau, assuming one 4 KiB page per row
        "required_iops": (48 / (tau * 1e-6)) if tau > 0 else float("inf"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, required=True, help="N, transformer layers")
    ap.add_argument(
        "--modules",
        required=True,
        help="comma list of layer:fetch_us pairs, e.g. '1:220,14:220'",
    )
    ap.add_argument("--tok-per-s", type=float, default=100.0)
    ap.add_argument(
        "--overhead-us",
        type=float,
        default=0.0,
        help="O: non-compute step overhead (scheduler/sampling/Python/H2D). "
        "0 assumes CUDA-graph-level efficiency; a real eager engine is >>0.",
    )
    ap.add_argument(
        "--compute-fraction",
        type=float,
        default=1.0,
        help="C / step. 1.0 means O is the only non-compute cost.",
    )
    ap.add_argument("--sweep", default=None, help="comma list of tok/s targets")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    mods = []
    for item in args.modules.split(","):
        layer_s, fetch_s = item.split(":")
        mods.append((int(layer_s), float(fetch_s)))

    targets = [args.tok_per_s]
    if args.sweep:
        targets = [float(x) for x in args.sweep.split(",")]

    out = {
        "layers": args.layers,
        "overhead_us": args.overhead_us,
        "compute_fraction": args.compute_fraction,
        "budget_us": BUDGET_US,
        "cases": [],
    }

    for tps in targets:
        step_us = 1e6 / tps
        compute_us = step_us * args.compute_fraction - args.overhead_us
        if compute_us < 0:
            raise SystemExit(
                f"tok/s={tps}: overhead {args.overhead_us}us exceeds step {step_us:.0f}us"
            )
        rows = []
        for layer, fetch in mods:
            rows.append(
                evaluate(
                    layers=args.layers,
                    layer=layer,
                    fetch_us=fetch,
                    step_us=step_us,
                    overhead_us=args.overhead_us,
                    compute_us=compute_us,
                )
            )
        out["cases"].append(
            {"tok_per_s": tps, "step_us": step_us, "compute_us": compute_us, "modules": rows}
        )

    # ---- report ---------------------------------------------------------- #
    print(
        f"N={args.layers} layers | O={args.overhead_us:.0f}us | "
        f"budget={BUDGET_US:.0f}us/token"
    )
    for case in out["cases"]:
        print(
            f"\n== target {case['tok_per_s']:.0f} tok/s  "
            f"(step {case['step_us']:.0f}us, compute {case['compute_us']:.0f}us) =="
        )
        print(
            f"{'layer':>5} {'fetch':>8} {'tau':>9} {'hidden':>7} "
            f"{'exposed':>9} {'%step':>7} {'%budget':>8} {'need IOPS':>11}"
        )
        for m in case["modules"]:
            print(
                f"{m['layer']:>5} {m['fetch_us']:>7.1f}u {m['tau_us']:>8.1f}u "
                f"{'YES' if m['fully_hidden'] else 'NO':>7} "
                f"{m['exposed_us']:>8.1f}u {m['exposed_pct_of_step']:>6.2f}% "
                f"{m['budget_pct']:>7.1f}% {m['required_iops']/1000:>9.0f}K"
            )

    if args.json_out:
        from pathlib import Path

        Path(args.json_out).write_text(json.dumps(out, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
