#!/usr/bin/env python3
"""End-to-end overhead budget check (Phase 2 closure artifact).

Two independent pieces of evidence are combined here, and neither alone is an
end-to-end claim:

1. **Component cost** (`probes/serving_ab_v041.json`): the real-table
   ``PleMemoryAdapter`` cost in microseconds per token, measured warm with
   random rowids across the whole table.  This is the number that actually
   enters the model's per-token budget.
2. **Integration proxy** (`probes/decode_ab_proxy_baseline.csv`): a real
   ``model.generate`` decode loop with a disk-backed embedding table swapped in.
   It proves the disk path plugs in and runs, and gives a *lower bound* on the
   cost -- the proxy reads **one** row per token (the input embedding), whereas
   Qwen PLE reads 16 and V4.1 Engram reads 48.

The derived relative overhead is therefore::

    overhead = engram_us_per_token / (1e6 / target_tokens_per_second)

which is a derivation from a measured component, not a measured end-to-end
model run.  It is reported as such: this closes the *proxy* loop and states the
precondition for the real-machine claim (see README section 3.2).

Usage:
    python scripts/overhead_budget_check.py
    python scripts/overhead_budget_check.py --serving-probe probes/serving_ab_v041.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

# Derived 5% budget; see docs/design.md section 7.3 and roadmap section 29.1.1.
TARGET_TOK_S = (50.0, 100.0)
MAX_OVERHEAD = 0.05
# Derived 5% budget in microseconds per token (docs/design.md section 7.3).
MAX_US_PER_TOKEN = 500.0
# Loose integration bound for the 1-row/token proxy (catches order-of-magnitude
# regressions; the proxy is far too small to gate tightly).
MAX_PROXY_RAW_SLOWDOWN = 0.25
ROWS_PER_TOKEN = {"qwen-ple": 16, "v41-engram": 48, "proxy-input-embedding": 1}
# The shipping target: only this one can fail the cold budget.
DEFAULT_PRODUCT = "qwen-ple"


def _load_adapter_us_per_token(path: Path) -> tuple[float | None, dict]:
    if not path.exists():
        return None, {}
    data = json.loads(path.read_text(encoding="utf-8"))
    adapter = data.get("ple_memory_adapter") or {}
    if "error" in adapter:
        raise SystemExit(f"FAIL: adapter path raised in {path.name}: {adapter['error']}")
    us = adapter.get("microseconds_per_token")
    if us is None and adapter.get("tokens_per_s"):
        us = 1e6 / adapter["tokens_per_s"]
    return us, data


def _load_proxy_slowdowns(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    out: dict[str, float] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            label = row.get("label", "")
            if not label.startswith("disk-"):
                continue
            raw = row.get("raw_slowdown") or row.get("lru_slowdown") or ""
            try:
                out[label] = float(raw)
            except ValueError:
                continue
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--serving-probe", default="probes/serving_ab_v041.json")
    ap.add_argument("--proxy-csv", default="probes/decode_ab_proxy_baseline.csv")
    ap.add_argument("--max-overhead", type=float, default=MAX_OVERHEAD)
    args = ap.parse_args()

    failures: list[str] = []
    skipped: list[str] = []

    print("[overhead-budget] --- component cost (real table) ---")
    us_per_token, data = _load_adapter_us_per_token(Path(args.serving_probe))
    if us_per_token is None:
        skipped.append(f"component cost ({args.serving_probe} missing)")
        print(f"  SKIP: {args.serving_probe} not present (run scripts/real_perf_gate.py)")
    else:
        print(
            f"  adapter = {us_per_token:.2f} us/token "
            f"({data.get('rowids')} rowids, warmup={data.get('warmup')}, reps={data.get('reps')})"
        )
        for name, rows in ROWS_PER_TOKEN.items():
            if name == "proxy-input-embedding":
                continue
            scaled = us_per_token * rows / ROWS_PER_TOKEN["qwen-ple"]
            for target in TARGET_TOK_S:
                budget_us = 1e6 / target
                overhead = scaled / budget_us
                status = "OK  " if overhead <= args.max_overhead else "FAIL"
                extra = ""
                if name == "v41-engram":
                    extra = "  (scaled 3x for 48 rows/token)"
                print(
                    f"  {status} {name:20s} @ {target:5.0f} tok/s -> "
                    f"{overhead:6.3%} of per-token time "
                    f"({scaled:.1f} us vs {budget_us:.0f} us budget){extra}"
                )
                if overhead > args.max_overhead:
                    failures.append(f"{name} @ {target:.0f} tok/s overhead {overhead:.2%}")

    print("[overhead-budget] --- integration proxy (1 row/token, lower bound) ---")
    slowdowns = _load_proxy_slowdowns(Path(args.proxy_csv))
    if not slowdowns:
        skipped.append(f"integration proxy ({args.proxy_csv} missing)")
        print(f"  SKIP: {args.proxy_csv} not present")
    else:
        for label, value in sorted(slowdowns.items()):
            status = "OK  " if value <= MAX_PROXY_RAW_SLOWDOWN else "FAIL"
            note = ""
            if value <= 0.0:
                note = "  (disk measured >= memory: noise floor, clamped)"
            print(f"  {status} {label:24s} slowdown={value:.1%}{note}")
            if value > MAX_PROXY_RAW_SLOWDOWN:
                failures.append(f"proxy {label} slowdown {value:.1%}")

    # --- cold (from-medium) read budget -------------------------------------
    # The warm numbers above measure the code path; these measure the medium,
    # and they are the ones that decide whether serving fits the budget at all.
    # On the real table over USB, cold reads run ~13-23x the warm ones, so a
    # warm-only gate reports a margin that serving does not have.
    print("[overhead-budget] --- cold (never-read rows) vs 5% budget ---")
    if us_per_token is None or not data.get("ple_memory_adapter"):
        skipped.append("cold read budget (needs a fresh real-table probe)")
        print("  SKIP: needs a probe with cold metrics")
    else:
        adapter = data["ple_memory_adapter"]
        cold_us = adapter.get("cold_microseconds_per_token")
        warm_us = adapter.get("microseconds_per_token")
        if cold_us is None:
            skipped.append("cold read budget (probe predates cold metrics)")
            print("  SKIP: probe has no cold_microseconds_per_token; re-run the bench")
        else:
            budget_us = MAX_US_PER_TOKEN
            ratio = cold_us / warm_us if warm_us else float("nan")
            print(
                f"  adapter cold = {cold_us:.1f} us/token, warm = {warm_us:.2f} us/token "
                f"({ratio:.1f}x)"
            )
            for name, rows in ROWS_PER_TOKEN.items():
                if name == "proxy-input-embedding":
                    continue
                scaled = cold_us * rows / ROWS_PER_TOKEN["qwen-ple"]
                frac = scaled / budget_us
                if frac > 1.0:
                    status, note = "FAIL", "  OVER BUDGET"
                elif frac > 0.5:
                    status, note = "WARN", "  thin margin (<2x headroom)"
                else:
                    status, note = "OK  ", ""
                extra = "  (projection, not measured)" if name != DEFAULT_PRODUCT else ""
                print(
                    f"  {status} {name:20s} cold @ {min(TARGET_TOK_S):.0f} tok/s -> "
                    f"{scaled:7.0f} us/token = {frac:6.0%} of {budget_us:.0f} us budget"
                    f"{note}{extra}"
                )
                # Only the shipping target can fail the gate.  V4.1 is a scaled
                # projection for a deferred target (roadmap 29.9): reported so
                # the 48-row risk stays visible, not to block unrelated work.
                if frac > 1.0 and name == DEFAULT_PRODUCT:
                    failures.append(
                        f"{name} cold {scaled:.0f} us/token over {budget_us:.0f} us budget"
                    )
            # The cold numbers above are a property of *this host's medium*, not of
            # the engine: the same code path measured 477 us on a USB HDD and
            # 204 us/token on native NVMe for the identical Qwen 16-row workload.
            # Do not read the v41-engram FAIL as device-independent -- it is a
            # scaling of whatever medium this host has.  See roadmap section 31.4
            # for the measured native-NVMe table (V4.1 = 604 us @8 threads, 353 us
            # @16, 223 us @32) and section 31.5 for the retracted 40x claim.
            print(
                "  note: cold figures are the MEDIUM's, not the engine's "
                "(this host's probe). Native-NVMe measurements: roadmap section 31.4"
            )

    if failures:
        print("[overhead-budget] FAIL: " + "; ".join(failures), file=sys.stderr)
        return 1
    if skipped:
        # Exit 0 keeps CI usable without the 51 GB real table, but a skip must
        # never be mistaken for evidence: the overhead claim is unverified.
        print(
            "[overhead-budget] SKIPPED (exit 0 for CI): "
            + "; ".join(skipped)
            + " -- overhead claim NOT verified",
            file=sys.stderr,
        )
        return 0
    print("[overhead-budget] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
