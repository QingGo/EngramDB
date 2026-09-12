#!/usr/bin/env python3
"""Serving-path A/B micro-benchmark for EngramDB.

This compares the low-level Store.fetch path, the PleMemory raw path, and the
optional torch PleMemoryAdapter path on either a synthetic Store or the real
Store-I tree (when ENGRAMDB_REAL_ROWS points at it).

Usage:
    python scripts/bench_serving_ab.py --synthetic --tokens 4096 --json-out /tmp/ab.json
    ENGRAMDB_REAL_ROWS=/path/to/real-rows python scripts/bench_serving_ab.py --tokens 4096
"""

from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
import time
from pathlib import Path

import engramdb


def _make_store(root: Path, rows: int, width: int = 160) -> engramdb.Store:
    with open(root / "shard_000.bin", "wb") as f:
        chunk = bytes(range(256)) * (width // 256 + 1)
        row = chunk[:width]
        for _ in range(rows):
            f.write(row)
    return engramdb.Store(str(root), 1, rows, width)


def _make_rowids(
    tokens: int,
    heads: int,
    rows_available: int,
    mode: str,
    seed: int,
    base_offset: int,
) -> list[int]:
    """Build the flat rowid list the benchmark reads.

    ``sequential`` reproduces the historical behaviour (rowids 0..tokens*heads),
    which touches only the first ~10 MB of the table and therefore never
    exercises the disk.  ``random`` spreads rowids across the whole table.

    ``base_offset`` exists so a run can be pointed at rows that have **never
    been read before**.  This matters more than it looks: repeated runs with a
    fixed seed re-read the same rows, which then live in the OS page cache, so
    the benchmark silently reports memory-speed numbers.  Measured on the real
    51.2 GB table over USB:

        serial, never-read rows   32.0 us/row
        serial, same rows again    1.9 us/row

    i.e. a 17x difference with no code change.  Cold is the number that matters
    for serving; warm is the number that matters for the code path.
    """
    total = tokens * heads
    if mode == "sequential":
        return [(base_offset + i) % rows_available for i in range(total)]
    rng = random.Random(seed)
    return [(base_offset + rng.randrange(rows_available)) % rows_available for _ in range(total)]


def _measure(fn, *, warmup: int, reps: int) -> dict:
    """Time ``fn`` reporting the **cold** first call and the warm median.

    The cold sample is the first invocation, before any warm-up.  The caller is
    responsible for pointing ``fn`` at never-before-read rows if it wants that
    number to mean "from the medium".
    """
    t0 = time.perf_counter()
    fn()
    cold = time.perf_counter() - t0

    for _ in range(max(0, warmup - 1)):
        fn()
    samples = []
    for _ in range(max(1, reps)):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    samples.sort()
    return {
        "seconds_cold": cold,
        "seconds_median": samples[len(samples) // 2],
        "seconds_min": samples[0],
        "seconds_max": samples[-1],
        "reps": len(samples),
        "warmup": warmup,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--base-offset",
        type=int,
        default=None,
        help="rowid base offset; default is a fresh random offset for the real "
        "table so rows are never-before-read (cold), 0 for synthetic",
    )
    parser.add_argument(
        "--rowids",
        choices=("sequential", "random"),
        default=None,
        help="rowid distribution; defaults to 'random' for the real table, "
        "'sequential' for synthetic",
    )
    args = parser.parse_args()

    real = os.environ.get("ENGRAMDB_REAL_ROWS")
    if not args.synthetic and real:
        store = engramdb.Store(real, 128, 2_500_012, 160)
        rows_available = 128 * 2_500_012
    else:
        td = tempfile.TemporaryDirectory(prefix="engramdb-serving-ab-")
        store = _make_store(Path(td.name), args.rows)
        rows_available = args.rows

    rowid_mode = args.rowids or ("random" if (real and not args.synthetic) else "sequential")
    if args.base_offset is not None:
        base_offset = args.base_offset
    elif real and not args.synthetic:
        base_offset = int.from_bytes(os.urandom(4), "big") % max(1, rows_available // 2)
    else:
        base_offset = 0

    from engramdb.ple_memory import PleMemory

    if real and not args.synthetic:
        mem = PleMemory(
            store=store,
            head_dim=160,
            num_heads=args.heads,
            ngram_size=3,
            heads_per_ngram=8,
        )
    else:
        # Synthetic rowid space: keep generated rowids inside the tiny shard.
        mem = PleMemory(
            store=store,
            head_dim=160,
            num_heads=args.heads,
            ngram_size=3,
            heads_per_ngram=8,
            prime_sizes=[2] * args.heads,
            offsets=list(range(args.heads)),
            eos=0,
        )
    try:
        tokens = args.tokens
        heads = args.heads
        # Each path gets its OWN disjoint slice of the rowid space.  Sharing one
        # rowid set would make only the first-measured path cold: it pulls those
        # rows into the page cache and every later path measures memory instead.
        stride = max(1, tokens * heads * 2)
        span = max(1, rows_available - stride)
        offsets = {
            name: (base_offset + i * stride) % span
            for i, name in enumerate(("store", "mem", "adapter"))
        }

        flat = _make_rowids(
            tokens, heads, rows_available, rowid_mode, args.seed, offsets["store"]
        )
        flat_mem = _make_rowids(
            tokens, heads, rows_available, rowid_mode, args.seed + 1, offsets["mem"]
        )
        rows_mem = [flat_mem[i * heads:(i + 1) * heads] for i in range(tokens)]

        stats_store = _measure(lambda: store.fetch(flat), warmup=args.warmup, reps=args.reps)
        stats_mem = _measure(
            lambda: mem.fetch_raw(rows_mem), warmup=args.warmup, reps=args.reps
        )
        dt_store = stats_store["seconds_median"]
        dt_mem = stats_mem["seconds_median"]

        adapter_result = None
        try:
            import torch
            from engramdb.adapter import PleMemoryAdapter

            adapter = PleMemoryAdapter(mem, keep_steps=0)
            n_adapter = min(tokens, 1000)
            # Shift token ids into the adapter's own slice so its first call is
            # genuinely cold too.
            ids = torch.tensor(
                list(range(offsets["adapter"], offsets["adapter"] + n_adapter)),
                dtype=torch.long,
            )

            def run_adapter():
                # A fresh sequence per call mirrors a new request; without the
                # reset the sequence accumulates tokens and stops being a
                # per-request measurement.
                adapter.reset()
                adapter(ids)

            stats_adapter = _measure(run_adapter, warmup=args.warmup, reps=args.reps)
            dt_adapter = stats_adapter["seconds_median"]
            cold_adapter = stats_adapter["seconds_cold"]
            adapter_result = {
                "seconds": dt_adapter,
                "tokens_per_s": n_adapter / dt_adapter if dt_adapter > 0 else None,
                "microseconds_per_token": (
                    dt_adapter * 1e6 / n_adapter if dt_adapter > 0 else None
                ),
                "cold_seconds": cold_adapter,
                "cold_microseconds_per_token": (
                    cold_adapter * 1e6 / n_adapter if cold_adapter > 0 else None
                ),
                "stats": stats_adapter,
            }
        except Exception as exc:
            adapter_result = {"error": repr(exc)}

        result = {
            "mode": "real" if real and not args.synthetic else "synthetic",
            "tokens": tokens,
            "heads": heads,
            "rowids": rowid_mode,
            "base_offset": base_offset,
            "path_offsets": offsets,
            "warmup": args.warmup,
            "reps": args.reps,
            "store_fetch_seconds": dt_store,
            "store_fetch_tokens_per_s": tokens / dt_store if dt_store > 0 else None,
            "store_fetch_microseconds_per_token": (
                dt_store * 1e6 / tokens if dt_store > 0 else None
            ),
            "store_fetch_cold_seconds": stats_store["seconds_cold"],
            "store_fetch_cold_microseconds_per_token": (
                stats_store["seconds_cold"] * 1e6 / tokens if tokens else None
            ),
            "store_fetch_stats": stats_store,
            "ple_memory_seconds": dt_mem,
            "ple_memory_tokens_per_s": tokens / dt_mem if dt_mem > 0 else None,
            "ple_memory_microseconds_per_token": (
                dt_mem * 1e6 / tokens if dt_mem > 0 else None
            ),
            "ple_memory_cold_seconds": stats_mem["seconds_cold"],
            "ple_memory_cold_microseconds_per_token": (
                stats_mem["seconds_cold"] * 1e6 / tokens if tokens else None
            ),
            "ple_memory_stats": stats_mem,
            "ple_memory_adapter": adapter_result,
        }
        print(json.dumps(result, indent=2, ensure_ascii=False))
        if args.json_out:
            Path(args.json_out).write_text(
                json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        return 0
    finally:
        store.close()
        if not real:
            td.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
