#!/usr/bin/env python3
"""Where does the per-call cost of a PLE read actually go?

The serving A/B (probes/serve_ple_ab_session42.md) measured ~913 us per call for
16 rows on the real table.  This decomposes that number so the optimisation
target is a measurement, not a guess.

Phases, in the order the reader executes them:

  1. rowid generation   engramdb.rowids_for_seq(tokens, PLE_QWEN_V1)   [PyO3]
  2. rowid folding      Python: modulo into the available row space
  3. Store.fetch        engramdb.Store.fetch(flat)                     [PyO3]
  4. tensor build       torch.frombuffer + view + reshape              [Python]

Phases 2 and 4 are pure Python and therefore hold the GIL.  If they dominate,
prefetching on a background thread cannot help (see docs/prefetch-lead-time.md
§6.2) and the fix is to collapse 1-4 into ONE PyO3 call under one
`py.allow_threads`.

Usage
-----
    python scripts/ple_reader_decompose.py --rows-dir /path/to/rows --reps 200
"""

from __future__ import annotations

import argparse
import statistics as st
import time
from pathlib import Path

SHARD_BYTES = 400_001_920
ROW_WIDTH = 160


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows-dir", required=True)
    ap.add_argument("--rows-per-token", type=int, default=16)
    ap.add_argument("--tokens", type=int, default=1, help="tokens per call")
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    import engramdb
    import torch

    shards = sorted(Path(args.rows_dir).glob("shard_*.bin"))
    if not shards:
        raise SystemExit(f"no shards under {args.rows_dir}")
    rows_total = len(shards) * (SHARD_BYTES // ROW_WIDTH)
    store = engramdb.Store(
        args.rows_dir, len(shards), SHARD_BYTES // ROW_WIDTH, ROW_WIDTH
    )
    print(
        f"table: {len(shards)} shards, {rows_total:,} rows, "
        f"{len(shards) * SHARD_BYTES / 2**30:.1f} GiB, width {ROW_WIDTH} B"
    )
    print(f"call shape: {args.tokens} token(s) x {args.rows_per_token} rows\n")

    import random

    rng = random.Random(args.seed)
    vocab = 200_000
    batches = [
        [rng.randrange(vocab) for _ in range(args.tokens)] for _ in range(args.reps)
    ]

    t_rowids, t_fold, t_fetch, t_build = [], [], [], []

    for toks in batches:
        t0 = time.perf_counter()
        ids = engramdb.rowids_for_seq(toks, engramdb.PLE_QWEN_V1)
        t1 = time.perf_counter()
        flat = []
        for row in ids:
            r = [int(x) % rows_total for x in row]
            if len(r) < args.rows_per_token:
                r = (r * (args.rows_per_token // len(r) + 1))[: args.rows_per_token]
            flat.extend(r[: args.rows_per_token])
        t2 = time.perf_counter()
        raw = store.fetch(flat)
        t3 = time.perf_counter()
        _ = (
            torch.frombuffer(bytearray(raw), dtype=torch.uint8)
            .view(torch.float32)
            .reshape(len(toks), -1)
        )
        t4 = time.perf_counter()
        t_rowids.append(t1 - t0)
        t_fold.append(t2 - t1)
        t_fetch.append(t3 - t2)
        t_build.append(t4 - t3)

    store.close()

    def rep(name: str, xs: list[float]) -> float:
        med = st.median(xs) * 1e6
        p90 = sorted(xs)[int(len(xs) * 0.9)] * 1e6
        print(f"{name:<24} median {med:>9.1f} us   p90 {p90:>9.1f} us")
        return med

    print("=== per-call decomposition ===")
    a = rep("1 rowid gen (PyO3)", t_rowids)
    b = rep("2 rowid fold (Python)", t_fold)
    c = rep("3 Store.fetch (PyO3)", t_fetch)
    d = rep("4 tensor build (Python)", t_build)
    tot = a + b + c + d
    print(f"{'TOTAL':<24} median {tot:>9.1f} us")
    print()
    print("=== share ===")
    for name, v in (
        ("rowid gen", a),
        ("rowid fold", b),
        ("fetch (I/O)", c),
        ("tensor build", d),
    ):
        print(f"  {name:<16} {v:>8.1f} us  {100 * v / tot:>5.1f}%")
    py = a + b + d
    print(
        f"\nGIL-holding Python phases (1+2+4): {py:.1f} us = {100 * py / tot:.1f}%\n"
        f"I/O phase (3):                     {c:.1f} us = {100 * c / tot:.1f}%"
    )
    print(
        "\nIf the Python share dominates, a background-thread prefetch cannot "
        "hide this cost -- it would contend for the GIL, which is what "
        "probes/serve_ple_ab_session42.md measured (2.7x worse)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
