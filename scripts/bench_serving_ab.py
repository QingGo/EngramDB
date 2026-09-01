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


def _timed(fn, tokens: int, heads: int) -> float:
    # Warm-up is deliberately included; this is a path latency comparison.
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--rows", type=int, default=100_000)
    args = parser.parse_args()

    real = os.environ.get("ENGRAMDB_REAL_ROWS")
    if not args.synthetic and real:
        store = engramdb.Store(real, 128, 2_500_012, 160)
        rows_available = 128 * 2_500_012
    else:
        td = tempfile.TemporaryDirectory(prefix="engramdb-serving-ab-")
        store = _make_store(Path(td.name), args.rows)
        rows_available = args.rows

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
        # Use rowids in-bounds (for synthetic they are 0..rows-1; for real use first block).
        flat = [i % rows_available for i in range(tokens * heads)]
        rows = [flat[i * heads:(i + 1) * heads] for i in range(tokens)]

        dt_store = _timed(lambda: store.fetch(flat), tokens, heads)
        dt_mem = _timed(lambda: mem.fetch_raw(rows), tokens, heads)

        adapter_result = None
        try:
            import torch
            from engramdb.adapter import PleMemoryAdapter

            adapter = PleMemoryAdapter(mem, keep_steps=0)
            ids = torch.tensor(list(range(min(tokens, 1000))), dtype=torch.long)
            dt_adapter = _timed(lambda: adapter(ids), tokens, heads)
            adapter_result = {
                "seconds": dt_adapter,
                "tokens_per_s": min(tokens, 1000) / dt_adapter if dt_adapter > 0 else None,
            }
        except Exception as exc:
            adapter_result = {"error": repr(exc)}

        result = {
            "mode": "real" if real and not args.synthetic else "synthetic",
            "tokens": tokens,
            "heads": heads,
            "store_fetch_seconds": dt_store,
            "store_fetch_tokens_per_s": tokens / dt_store if dt_store > 0 else None,
            "ple_memory_seconds": dt_mem,
            "ple_memory_tokens_per_s": tokens / dt_mem if dt_mem > 0 else None,
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
