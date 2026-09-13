#!/usr/bin/env python3
"""Phase-level micro-benchmark for the graph-mode disk-read path.

Why this exists (roadmap §36.5, P1.5 prerequisite)
--------------------------------------------------
Session 43 measured the graph-mode break by *differencing arms* and read
"disk and D2H are about half each" out of a ±0.2 ms noise band.  That was wrong,
and the reason is methodological: a difference of two noisy totals cannot
attribute cost.  The fix is to time each phase **directly**, in place, and then
compare those numbers against the same phases measured **outside the engine**.

That comparison is the whole point of this script.  A phase can be expensive for
two very different reasons:

* **intrinsic** -- the work itself costs that much (e.g. 16 page faults on NVMe)
* **critical-path amplified** -- the work is cheap alone but expensive inside a
  decode step, because it drains a queued GPU pipeline before it can start

Only the out-of-band number can tell them apart, and they need opposite fixes.
So every phase below is measured with the GPU idle, and `--busy` re-measures the
sync-sensitive ones with a long kernel queued, which is the condition inside a
graph break.

Phases
------
``d2h_1``        ``.tolist()`` on a 1-element CUDA int64 tensor
``rowid_1``      ``ref_ple_hash.ple_rowids`` for T=1
``rowid_32``     ... for T=32  (amortisation of the fixed numpy overhead)
``read_16``      16 rows x 160 B, cold, one token, swept over concurrency
``read_512``     512 rows (16 rows x 32 tokens) in one call -- the batched form
``h2d_1024``     1024 float32 host -> device, pageable vs pinned
``add_1024``     ``tensor.add_(delta)`` on device

Usage
-----
    python scripts/sc4_phase_microbench.py \\
        --store /root/autodl-tmp/qwen35-ple/qwen38-rows \\
        --json-out probes/sc4_phase_microbench.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROWS_PER_SHARD = 2_500_012
ROW_BYTES = 160
N_SHARDS = 128
N_ROWS = 320_001_536


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------

class Samples:
    def __init__(self, name: str, unit: str = "us"):
        self.name, self.unit, self.v = name, unit, []

    def add(self, seconds: float) -> None:
        self.v.append(seconds * 1e6)

    def summary(self) -> dict:
        v = sorted(self.v)
        if not v:
            return {"name": self.name, "n": 0}
        return {
            "name": self.name, "n": len(v), "unit": self.unit,
            "min": round(v[0], 2),
            "median": round(statistics.median(v), 2),
            "p90": round(v[int(0.9 * (len(v) - 1))], 2),
            "max": round(v[-1], 2),
            "mean": round(sum(v) / len(v), 2),
        }


# ---------------------------------------------------------------------------
# reader (standalone on purpose: the instrument must not share code with the
# thing it measures, or a bug in the reader hides itself)
# ---------------------------------------------------------------------------

class Reader:
    def __init__(self, store: str, workers: int = 64):
        self.store = store
        self.fds = [os.open(os.path.join(store, f"shard_{i:03d}.bin"), os.O_RDONLY)
                    for i in range(N_SHARDS)]
        self.pool = ThreadPoolExecutor(max_workers=workers)
        self.workers = workers

    def fadvise_shards(self, shards) -> None:
        for s in set(shards):
            os.posix_fadvise(self.fds[s], 0, 0, os.POSIX_FADV_DONTNEED)

    def read(self, rowids, chunksize: int, pool: ThreadPoolExecutor | None = None):
        """rowids: flat iterable of ints.  Returns bytes read."""
        pool = pool or self.pool
        buf = bytearray(len(rowids) * ROW_BYTES)

        def one(job):
            idx, rowid = job
            shard, off = divmod(rowid, ROWS_PER_SHARD)
            return idx, os.pread(self.fds[shard], ROW_BYTES, off * ROW_BYTES)

        jobs = ((i, int(r)) for i, r in enumerate(rowids))
        for idx, data in pool.map(one, jobs, chunksize=chunksize):
            buf[idx * ROW_BYTES:(idx + 1) * ROW_BYTES] = data
        return len(buf)


def random_rowids(n: int, seed: int):
    import random
    rng = random.Random(seed)
    return [rng.randrange(N_ROWS) for _ in range(n)]


# ---------------------------------------------------------------------------
# phases
# ---------------------------------------------------------------------------

def bench_d2h(rows: int, busy: bool) -> list[Samples]:
    import torch

    dev = "cuda"
    tag = "_gpu_busy" if busy else "_gpu_idle"
    out = []
    for label, tensor_fn in ((f"d2h_1{tag}", lambda: torch.zeros(1, dtype=torch.int64, device=dev)),
                             (f"d2h_32{tag}", lambda: torch.zeros(32, dtype=torch.int64, device=dev))):
        s = Samples(label)
        t = tensor_fn()
        torch.cuda.synchronize()
        for _ in range(rows):
            if busy:
                # Queue a long kernel and DO NOT sync: this is the real condition
                # inside a graph break, where segment 0's kernels are still in
                # flight when .tolist() is called.  The first version of this
                # benchmark synced before t0, which drained exactly the queue the
                # busy case was supposed to create -- and so "proved" that D2H
                # does not wait, which it was never in a position to show.
                a = torch.randn(4096, 4096, device=dev)
                torch.matmul(a, a)
            t0 = time.perf_counter()
            _ = t.tolist()
            s.add(time.perf_counter() - t0)
            torch.cuda.synchronize()
        out.append(s)
    return out


def bench_rowid(rows: int) -> list[Samples]:
    import numpy as np
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from ref_ple_hash import head_vocab_sizes, ple_rowids, spec_from_weights

    docs = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs")
    mult = spec_from_weights(docs)
    sizes = head_vocab_sizes()

    out = []
    for T in (1, 8, 32, 128):
        toks = np.array([1000 + (7 * i) % 200000 for i in range(T)], dtype=np.int64)
        s = Samples(f"rowid_{T}")
        for _ in range(rows):
            t0 = time.perf_counter()
            ids = ple_rowids(toks, mult, sizes)
            s.add(time.perf_counter() - t0)
        s.v = s.v[3:]               # drop warmup (import/first-call effects)
        out.append(s)
        assert ids.shape == (T, 16)
    return out


def bench_read(reader: Reader, rows: int, cold: bool) -> list[Samples]:
    out = []
    # 1) single token (16 rows), swept over pool width and chunksize
    for workers, chunksize in ((1, 1), (4, 1), (16, 1), (64, 1), (64, 8), (64, 16)):
        pool = ThreadPoolExecutor(max_workers=workers)
        # Warm the pool: ThreadPoolExecutor spawns threads lazily on submit, so
        # without this the first `workers` reps pay thread creation and the
        # median is a measurement of thread startup, not of I/O.
        warm = random_rowids(16 * max(workers, 8), seed=999)
        reader.fadvise_shards([r // ROWS_PER_SHARD for r in warm])
        reader.read(warm, chunksize=chunksize, pool=pool)
        s = Samples(f"read_16_tok_w{workers}_c{chunksize}")
        for r in range(rows):
            rids = random_rowids(16, seed=1000 + r)
            if cold:
                reader.fadvise_shards([rid // ROWS_PER_SHARD for rid in rids])
            t0 = time.perf_counter()
            reader.read(rids, chunksize=chunksize, pool=pool)
            s.add(time.perf_counter() - t0)
        pool.shutdown()
        out.append(s)

    # 2) batched: 32 tokens x 16 rows in ONE call -- what a real decode step with
    #    batching would issue
    for batch in (8, 32, 128):
        s = Samples(f"read_batch{batch}_tok")
        per_token = []
        for r in range(rows):
            rids = random_rowids(16 * batch, seed=5000 + r)
            if cold:
                reader.fadvise_shards([rid // ROWS_PER_SHARD for rid in rids])
            t0 = time.perf_counter()
            reader.read(rids, chunksize=1)
            dt = time.perf_counter() - t0
            s.add(dt)
            per_token.append(dt / batch)
        out.append(s)
        pt = Samples(f"read_batch{batch}_per_token")
        pt.v = [x * 1e6 for x in per_token]
        out.append(pt)
    return out


def bench_transfer_and_add(rows: int) -> list[Samples]:
    import numpy as np
    import torch

    dev = "cuda"
    host = np.zeros(1024, dtype=np.float32)
    host_pinned = torch.zeros(1024, dtype=torch.float32, pin_memory=True)
    dst = torch.zeros(1024, dtype=torch.float32, device=dev)
    out = []

    for label, fn in (
        ("h2d_1024_pageable", lambda: torch.from_numpy(host.copy()).to(dev)),
        ("h2d_1024_pinned_nonblocking",
         lambda: dst.copy_(host_pinned, non_blocking=True)),
    ):
        s = Samples(label)
        for _ in range(rows):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            s.add(time.perf_counter() - t0)
        out.append(s)

    src_dev = torch.zeros(1024, dtype=torch.float32, device=dev)
    s = Samples("add_1024_device")
    for _ in range(rows):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        dst.add_(src_dev)
        torch.cuda.synchronize()
        s.add(time.perf_counter() - t0)
    out.append(s)

    s = Samples("empty_1024_device_alloc")
    for _ in range(rows):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = torch.empty(1024, dtype=torch.float32, device=dev)
        torch.cuda.synchronize()
        s.add(time.perf_counter() - t0)
    out.append(s)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--rows", type=int, default=25, help="reps per configuration")
    ap.add_argument("--warm", action="store_true", help="do not drop page cache")
    ap.add_argument("--skip-read", action="store_true")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    cold = not args.warm
    print(f"store={args.store} rows={args.rows} cold={cold}")

    all_s: list[Samples] = []
    print("\n[1] D2H (.tolist()) -- GPU idle vs GPU busy")
    all_s += bench_d2h(args.rows, busy=False)
    all_s += bench_d2h(args.rows, busy=True)
    for s in all_s:
        print("   ", s.summary())

    print("\n[2] rowid computation (warm prime table)")
    r = bench_rowid(max(args.rows, 10))
    for s in r:
        print("   ", s.summary())
    all_s += r

    print("\n[3] transfer + add")
    t = bench_transfer_and_add(args.rows)
    for s in t:
        print("   ", s.summary())
    all_s += t

    if not args.skip_read:
        print("\n[4] disk read (this is the slow one)")
        reader = Reader(args.store)
        rd = bench_read(reader, args.rows, cold)
        for s in rd:
            print("   ", s.summary())
        all_s += rd

    print("\n=== SUMMARY (median us) ===")
    for s in all_s:
        d = s.summary()
        if d.get("n"):
            print(f"  {d['name']:<34} median={d['median']:>9.1f}  "
                  f"p90={d['p90']:>9.1f}  min={d['min']:>9.1f}  n={d['n']}")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(
            {"store": args.store, "rows": args.rows, "cold": cold,
             "phases": [s.summary() for s in all_s]}, indent=2) + "\n")
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
