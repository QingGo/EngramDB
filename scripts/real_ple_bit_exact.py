#!/usr/bin/env python3
"""Bit-exact check between the real PLE row files and EngramDB Store reads.

This validates the disk-facing store primitive against the exact bytes in the
extracted Qwen PLE shards.  It does not require loading the full model.

Example:

    python scripts/real_ple_bit_exact.py \
        --rows-dir data/real-rows \
        --n 1024
"""

from __future__ import annotations

import argparse
import hashlib
import random
from pathlib import Path

import engramdb


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows-dir", default="data/real-rows")
    ap.add_argument("--shards", type=int, default=128)
    ap.add_argument("--rows-per-shard", type=int, default=2_500_012)
    ap.add_argument("--width", type=int, default=160)
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    total = args.shards * args.rows_per_shard
    rng = random.Random(args.seed)
    rowids = [rng.randrange(total) for _ in range(args.n)]

    store = engramdb.Store(
        args.rows_dir,
        args.shards,
        args.rows_per_shard,
        args.width,
    )

    mismatches = 0
    raw_fnv = hashlib.sha256()
    store_fnv = hashlib.sha256()
    for rowid in rowids:
        shard = rowid // args.rows_per_shard
        offset = (rowid % args.rows_per_shard) * args.width
        path = Path(args.rows_dir) / f"shard_{shard:03d}.bin"
        with open(path, "rb") as f:
            f.seek(offset)
            expected = f.read(args.width)

        got = store.fetch([rowid])
        if got != expected:
            mismatches += 1
            if mismatches <= 5:
                print(f"MISMATCH rowid={rowid} shard={shard} offset={offset}")
        raw_fnv.update(expected)
        store_fnv.update(got)

    store.close()

    print(f"checked={args.n} rowids shards={args.shards} rows_per_shard={args.rows_per_shard} width={args.width}")
    print(f"raw sha256:  {raw_fnv.hexdigest()}")
    print(f"store sha256: {store_fnv.hexdigest()}")
    print(f"mismatches: {mismatches}")
    if mismatches == 0:
        print("PLE_STORE_BIT_EXACT_PASS")
    else:
        raise SystemExit("PLE_STORE_BIT_EXACT_FAIL")


if __name__ == "__main__":
    main()
