#!/usr/bin/env python3
"""Which host-side call in the staged read costs what, and can the API be cheaper?

The graph step is `d2h(ids) + store read + host bookkeeping + publish`. Phases
put `Store.fetch` at ~673 us for 2048 rows and the H2D at ~43, which leaves ~670
unattributed. This isolates the candidates -- id list conversion, the shard mask,
the bytes->tensor wrapper, the pinned copy -- and checks whether the binding will
take a numpy array or a `torch.Tensor` directly instead of a Python list, since
`.tolist()` on 2048 ids builds 2048 Python ints.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time

import numpy as np
import torch

DEFAULT_STORE = "/root/autodl-tmp/qwen35-ple/qwen38-rows"
ROWS, SHARDS, ROWS_PER_SHARD, WIDTH = 320_001_536, 128, 2_500_012, 160


def timed(fn, n=50):
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e6)
    return {"median": statistics.median(ts), "min": min(ts)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default=DEFAULT_STORE)
    ap.add_argument("--rows", default="16,128,512,2048")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import engramdb

    store = engramdb.Store(
        args.store, shards=SHARDS, rows_per_shard=ROWS_PER_SHARD, width=WIDTH, threads=32
    )
    report = {}
    for rows in [int(x) for x in args.rows.split(",")]:
        ids_t = torch.randint(0, ROWS, (rows,), dtype=torch.int64)
        lst = ids_t.tolist()
        arr = ids_t.numpy()
        raw = store.fetch(lst)
        row = {
            "tolist": timed(ids_t.tolist, args.iters),
            "numpy_view": timed(lambda: ids_t.numpy(), args.iters),
            "fetch_list": timed(lambda: store.fetch(lst), args.iters),
        }
        try:
            out = store.fetch(arr)
            row["fetch_numpy_ok"] = True
            row["fetch_numpy"] = timed(lambda: store.fetch(arr), args.iters)
            row["fetch_numpy_same"] = out == raw
        except Exception as exc:
            row["fetch_numpy_ok"] = False
            row["fetch_numpy_error"] = repr(exc)[:200]
        try:
            row["fetch_tensor_ok"] = True
            got = store.fetch(ids_t)
            row["fetch_tensor_same"] = got == raw
            row["fetch_tensor"] = timed(lambda: store.fetch(ids_t), args.iters)
        except Exception as exc:
            row["fetch_tensor_ok"] = False
            row["fetch_tensor_error"] = repr(exc)[:200]

        # The pieces of _read_rows after fetch returns.
        row["bytearray_copy"] = timed(lambda: bytearray(raw), args.iters)

        def wrap():
            return (
                torch.frombuffer(bytearray(raw), dtype=torch.uint8)
                .view(torch.float8_e4m3fn)
                .view(-1, WIDTH)
            )

        row["wrap_fp8"] = timed(wrap, args.iters)
        src = wrap()
        slab = torch.empty((rows, WIDTH), dtype=torch.float8_e4m3fn, pin_memory=True)
        row["pinned_memcpy"] = timed(lambda: slab.copy_(src), args.iters)
        mask = (ids_t >= 0) & (ids_t < ROWS)
        row["mask_and_all"] = timed(lambda: bool(((ids_t >= 0) & (ids_t < ROWS)).all()), args.iters)
        row["mask_scatter_pinned"] = timed(lambda: slab.__setitem__(mask, src), args.iters)
        report[f"rows={rows}"] = {
            k: (v if not isinstance(v, dict) else v["median"]) for k, v in row.items()
        }
        print(
            f"rows={rows:>5} " + "  ".join(f"{k}={v if not isinstance(v, dict) else round(v['median'],1)}"
                                           for k, v in row.items()),
            flush=True,
        )
    store.close()
    if args.out:
        json.dump(report, open(args.out, "w"), indent=2)
    print(json.dumps(report, indent=2)[:3000])


if __name__ == "__main__":
    main()
