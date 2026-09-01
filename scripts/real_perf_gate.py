#!/usr/bin/env python3
"""Real-table serving performance threshold gate.

Usage:
    ENGRAMDB_REAL_ROWS=/path/to/real-rows python scripts/real_perf_gate.py

If the real Store tree is not available, this script exits 0 and prints a skip
message (matching CI where real data is not present).
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
        ple_rps = data.get("ple_memory_tokens_per_s") or 0.0
        fetch_rps = data.get("store_fetch_tokens_per_s") or 0.0
        print(
            f"[real-perf-gate] ple_memory={ple_rps:.0f} rows/s "
            f"store_fetch={fetch_rps:.0f} rows/s"
        )
        if ple_rps < MIN_PLE_MEMORY_RPS or fetch_rps < MIN_STORE_FETCH_RPS:
            print(
                f"[real-perf-gate] FAIL: thresholds "
                f"ple>={MIN_PLE_MEMORY_RPS} store>={MIN_STORE_FETCH_RPS}"
            )
            return 1
        print("[real-perf-gate] OK")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
