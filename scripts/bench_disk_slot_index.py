#!/usr/bin/env python3
"""Benchmark the native/Python DiskSlotIndex path on real or synthetic keys.

This is the Phase B2 tool for WSL 10M/100M/320M validation.  It builds a
disk slot index from a flat EngramDB keys file (16 rowids per gram), verifies
it with the native CLI, and then measures Python lookup latency with a bounded
bucket LRU.

Usage:

    python scripts/bench_disk_slot_index.py \
        --keys /path/to/view.keys \
        --out /tmp/slot-idx \
        --buckets 16384 \
        --samples 10000 \
        --engramdb-bin target/release/engramdb

If --keys is omitted, a synthetic keys file is generated for --grams grams.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np


def write_synthetic_keys(path: Path, grams: int, heads: int = 16) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for g in range(grams):
            for h in range(heads):
                f.write(f"{g * heads + h}\n")


def run(cmd: list[str]) -> tuple[int, str, str]:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keys", default=None)
    parser.add_argument("--grams", type=int, default=100_000)
    parser.add_argument("--out", required=True)
    parser.add_argument("--buckets", type=int, default=16384)
    parser.add_argument("--cache", type=int, default=64)
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--engramdb-bin", default="engramdb")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    if args.keys:
        keys_path = Path(args.keys)
    else:
        td = tempfile.TemporaryDirectory(prefix="disk-slot-bench-keys-")
        keys_path = Path(td.name) / "keys.txt"
        write_synthetic_keys(keys_path, args.grams)
        keys_path = keys_path

    out = Path(args.out)
    print(f"[disk-slot-bench] keys={keys_path} out={out} buckets={args.buckets}")

    # Native build
    t0 = time.perf_counter()
    rc, stdout, stderr = run(
        [
            args.engramdb_bin,
            "slot-index",
            "build",
            str(keys_path),
            str(out),
            "--buckets",
            str(args.buckets),
        ]
    )
    build_s = time.perf_counter() - t0
    if rc != 0:
        print(stdout)
        print(stderr)
        raise SystemExit(f"slot-index build failed: {rc}")
    print(stdout.strip())

    # Native verify
    t0 = time.perf_counter()
    rc, stdout, stderr = run(
        [args.engramdb_bin, "slot-index", "verify", str(keys_path), str(out)]
    )
    verify_s = time.perf_counter() - t0
    if rc != 0:
        print(stdout)
        print(stderr)
        raise SystemExit(f"slot-index verify failed: {rc}")
    print(stdout.strip())

    # Python lookup latency
    import engramdb

    if engramdb.DiskSlotIndex is None:
        print("[disk-slot-bench] DiskSlotIndex unavailable (numpy missing); skipping Python lookups")
        lookup_result = None
    else:
        idx = engramdb.DiskSlotIndex.open(out, cache_buckets=args.cache)
        try:
            total = int(args.samples)
            # Build a sample matrix with the same simple rowid pattern.
            rows = np.arange(total * 16, dtype=np.int64).reshape(total, 16)
            t0 = time.perf_counter()
            slots = idx.to_slots(rows)
            lookup_s = time.perf_counter() - t0
            lookup_result = {
                "samples": total,
                "lookup_seconds": lookup_s,
                "lookups_per_s": total / lookup_s if lookup_s > 0 else None,
                "us_per_lookup": lookup_s / total * 1e6 if total else None,
                "stats": idx.stats(),
            }
            assert len(slots) == total
        finally:
            idx.close()

    result = {
        "keys": str(keys_path),
        "grams": sum(1 for _ in open(keys_path)) // 16,
        "buckets": args.buckets,
        "build_seconds": build_s,
        "verify_seconds": verify_s,
        "python_lookup": lookup_result,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
