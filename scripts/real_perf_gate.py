#!/usr/bin/env python3
"""Real-table serving performance threshold gate.

Usage:
    ENGRAMDB_REAL_ROWS=/path/to/real-rows python scripts/real_perf_gate.py

If the real Store tree is not available, this script exits 0 and prints a skip
message (matching CI where real data is not present).

Measurement discipline
----------------------
Every path is measured after warm-up as the **median of N runs** over rowids
drawn from the **whole table**.  Two earlier practice flaws made this gate
report numbers that were not reproducible:

1. A single un-warmed sample is dominated by first-touch page-cache misses on
   the external volume.  The same unchanged code measured 1.7K tok/s once and
   25.8K tok/s on the next run.
2. Rowids 0..tokens*heads touch only the first ~10 MB of a 51.2 GB table, so the
   "real table" path never exercised the disk at all.

Budget
------
At 50-100 tok/s end-to-end decode, a 5% overhead budget is 500-1000 us/token;
we gate on the conservative 500 us/token.  See docs/design.md section 7 for the
derivation.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REAL_ROWS = Path(os.environ.get("ENGRAMDB_REAL_ROWS", "data/real-rows"))

MIN_PLE_MEMORY_RPS = 5_000.0
MIN_STORE_FETCH_RPS = 5_000.0
# Phase 1 target from docs/roadmap.md section 29.4: adapter <= 50 us/token.
MIN_ADAPTER_RPS = 20_000.0
# Derived 5% budget (docs/roadmap.md section 29.1.1).
MAX_US_PER_TOKEN = 500.0


def _fail(*lines: str) -> int:
    for line in lines:
        print(line, file=sys.stderr)
    return 1


def _evaluate(data: dict) -> tuple[bool, list[str], dict]:
    """Decide PASS/FAIL from one bench result.

    Split out from :func:`main` so the decision logic is unit-testable without
    a real table (see ``scripts/real_perf_gate_test.py``).
    """
    ple_rps = data.get("ple_memory_tokens_per_s") or 0.0
    fetch_rps = data.get("store_fetch_tokens_per_s") or 0.0
    adapter = data.get("ple_memory_adapter") or {}
    info = {
        "ple_rps": ple_rps,
        "fetch_rps": fetch_rps,
        "adapter_rps": 0.0,
        "us_per_token": None,
        "rowids": data.get("rowids"),
        "warmup": data.get("warmup"),
        "reps": data.get("reps"),
    }

    if "error" in adapter:
        return False, [f"adapter path raised: {adapter['error']}"], info
    adapter_rps = adapter.get("tokens_per_s")
    if not adapter_rps:
        return False, [
            "ple_memory_adapter.tokens_per_s missing "
            "('not measured' is FAIL, not PASS)"
        ], info

    us_per_token = adapter.get("microseconds_per_token") or (1e6 / adapter_rps)
    info["adapter_rps"] = adapter_rps
    info["us_per_token"] = us_per_token

    failures = []
    if ple_rps < MIN_PLE_MEMORY_RPS:
        failures.append(f"ple_memory {ple_rps:,.0f} < {MIN_PLE_MEMORY_RPS:,.0f}")
    if fetch_rps < MIN_STORE_FETCH_RPS:
        failures.append(f"store_fetch {fetch_rps:,.0f} < {MIN_STORE_FETCH_RPS:,.0f}")
    if adapter_rps < MIN_ADAPTER_RPS:
        failures.append(f"adapter {adapter_rps:,.0f} < {MIN_ADAPTER_RPS:,.0f}")
    if us_per_token > MAX_US_PER_TOKEN:
        failures.append(f"adapter {us_per_token:.1f} > {MAX_US_PER_TOKEN:.0f} us/token")
    return (not failures), failures, info


def main() -> int:
    if not (REAL_ROWS / "shard_000.bin").exists():
        print("[real-perf-gate] skip: real Store not present")
        return 0

    with tempfile.TemporaryDirectory(prefix="engramdb-real-perf-") as td:
        json_path = Path(td) / "result.json"
        env = dict(os.environ)
        env["ENGRAMDB_REAL_ROWS"] = str(REAL_ROWS)
        proc = subprocess.run(
            [
                sys.executable,
                "scripts/bench_serving_ab.py",
                "--tokens",
                "4096",
                "--json-out",
                str(json_path),
            ],
            cwd=Path(__file__).resolve().parents[1],
            env=env,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            print(proc.stdout)
            print(proc.stderr, file=sys.stderr)
            return 1
        data = json.loads(json_path.read_text(encoding="utf-8"))

        ok, failures, info = _evaluate(data)
        print(
            f"[real-perf-gate] rowids={info['rowids']} "
            f"warmup={info['warmup']} reps={info['reps']}"
        )
        print(
            f"[real-perf-gate] ple_memory={info['ple_rps']:,.0f} tok/s  "
            f"store_fetch={info['fetch_rps']:,.0f} tok/s  "
            f"adapter={info['adapter_rps']:,.0f} tok/s"
        )
        if info["us_per_token"] is not None:
            print(
                f"[real-perf-gate] adapter={info['us_per_token']:.2f} us/token "
                f"(budget {MAX_US_PER_TOKEN:.0f} us/token)"
            )
        if not ok:
            return _fail("[real-perf-gate] FAIL: " + "; ".join(failures))
        print("[real-perf-gate] OK")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
