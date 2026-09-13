#!/usr/bin/env python3
"""Is `t_read = 445 us/token` a *disk* number or a *reader* number?

The question this answers
-------------------------
`scripts/sc4_phase_microbench.py` found that 16 cold random rows cost:

    python os.pread, 1 thread    1917 us
    python os.pread, 4 threads    627 us   (3.06x)
    python os.pread, 16 threads   604 us
    python os.pread, 64 threads   577 us   (3.32x -- saturated)
    batched to 128 tokens/token   464 us   (no better)

Saturation at ~3.3x while the device is 18x below its IOPS capability means the
limit is **not the disk**.  A serialised critical section in the dispatch path
looks exactly like this, and the prime suspect is the Python thread pool: 16
`concurrent.futures` round-trips under the GIL.

That matters a great deal, because `t_read` is the **numerator of `L*`** -- the
entire schedulability verdict rests on it.  If 445 us is a Python artifact and
the native reader reaches the ~196 us measured in earlier sessions, then the
verdict is about our reader, not about storage.

So: same rows, same coldness, both readers.

Usage
-----
    python scripts/sc4_reader_compare.py --store .../qwen38-rows \\
        --json-out probes/sc4_reader_compare.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROWS_PER_SHARD = 2_500_012
ROW_BYTES = 160
N_SHARDS = 128
N_ROWS = 320_001_536


class S:
    def __init__(self, name):
        self.name, self.v = name, []

    def add(self, sec):
        self.v.append(sec * 1e6)

    def summary(self):
        v = sorted(self.v)
        if not v:
            return {"name": self.name, "n": 0}
        return {"name": self.name, "n": len(v), "min": round(v[0], 1),
                "median": round(statistics.median(v), 1),
                "p90": round(v[int(0.9 * (len(v) - 1))], 1)}


def rids(n, seed):
    r = random.Random(seed)
    return [r.randrange(N_ROWS) for _ in range(n)]


def cold(fds, rowids):
    for s in {r // ROWS_PER_SHARD for r in rowids}:
        os.posix_fadvise(fds[s], 0, 0, os.POSIX_FADV_DONTNEED)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--rows", type=int, default=30)
    ap.add_argument("--warm", action="store_true")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()
    is_cold = not args.warm

    import engramdb

    fds = [os.open(os.path.join(args.store, f"shard_{i:03d}.bin"), os.O_RDONLY)
           for i in range(N_SHARDS)]
    out: list[S] = []

    def pyread(rowids, pool=None, chunksize=1):
        buf = bytearray(len(rowids) * ROW_BYTES)

        def one(job):
            i, rid = job
            sh, off = divmod(rid, ROWS_PER_SHARD)
            return i, os.pread(fds[sh], ROW_BYTES, off * ROW_BYTES)

        jobs = ((i, int(r)) for i, r in enumerate(rowids))
        if pool is None:
            for i, d in map(one, jobs):
                buf[i * ROW_BYTES:(i + 1) * ROW_BYTES] = d
        else:
            for i, d in pool.map(one, jobs, chunksize=chunksize):
                buf[i * ROW_BYTES:(i + 1) * ROW_BYTES] = d
        return len(buf)

    # ---- python readers -------------------------------------------------
    for w in (1, 4, 16, 64):
        pool = ThreadPoolExecutor(max_workers=w)
        warm = rids(16 * max(w, 8), 999)
        cold(fds, warm) if is_cold else None
        pyread(warm, pool)                      # warm the pool, not timed
        s = S(f"python_pread_pool{w}")
        for r in range(args.rows):
            rr = rids(16, 1000 + r)
            if is_cold:
                cold(fds, rr)
            t0 = time.perf_counter()
            pyread(rr, pool)
            s.add(time.perf_counter() - t0)
        pool.shutdown()
        out.append(s)

    # ---- native readers -------------------------------------------------
    def try_native(label, fn, per_rep_rowids=None):
        s = S(label)
        try:
            fn(rids(16, 7))                      # smoke
        except Exception as exc:
            print(f"  [{label}] unavailable: {type(exc).__name__}: {exc}")
            return None
        for r in range(args.rows):
            rr = per_rep_rowids(r)
            if is_cold:
                cold(fds, rr)
            t0 = time.perf_counter()
            fn(rr)
            s.add(time.perf_counter() - t0)
        out.append(s)
        return s

    print("[native] constructing Store ...")
    store = None
    for ctor in (
        lambda: engramdb.Store(args.store, N_SHARDS, ROWS_PER_SHARD, ROW_BYTES),
        lambda: engramdb.Store(args.store),
    ):
        try:
            store = ctor()
            print(f"  Store ok: total_rows={store.total_rows} width={store.width}")
            break
        except Exception as exc:
            print(f"  ctor failed: {type(exc).__name__}: {exc}")
    if store is not None:
        try_native("native_store_fetch16",
                   lambda rr: store.fetch(rr), lambda r: rids(16, 1000 + r))
        try_native("native_store_fetch_one16",
                   lambda rr: [store.fetch_one(x) for x in rr],
                   lambda r: rids(16, 1000 + r))

    print("[native] constructing StorePool ...")
    def pool_fn(rr, pool_size):
        p = engramdb.StorePool(args.store, N_SHARDS, ROWS_PER_SHARD, ROW_BYTES,
                               pool_size=pool_size)
        tls = engramdb.ThreadLocalStore(p)
        st = tls.get()
        try:
            return st.fetch(rr)
        finally:
            p.close()

    for ps in (4, 16):
        try_native(f"native_storepool{ps}_fetch16",
                   lambda rr, ps=ps: pool_fn(rr, ps),
                   lambda r: rids(16, 1000 + r))

    print("\n=== READER COMPARISON (us per 16-row token, cold=%s) ===" % is_cold)
    for s in out:
        d = s.summary()
        if d.get("n"):
            print(f"  {d['name']:<30} median={d['median']:>8.1f}  "
                  f"p90={d['p90']:>8.1f}  min={d['min']:>8.1f}")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(
            {"store": args.store, "cold": is_cold, "rows": args.rows,
             "readers": [s.summary() for s in out]}, indent=2) + "\n")
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
