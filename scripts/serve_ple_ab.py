#!/usr/bin/env python3
"""Real-engine serving A/B: EngramDB disk PLE reader vs in-RAM vs mmap.

Engine  vLLM 0.29.0 offline ``LLM`` API (real scheduler, KV cache, continuous
        batching, real sampling).  This is serving, not a micro-benchmark.
Model   Qwen3.5-0.8B (``Qwen3_5ForConditionalGeneration``, 24 layers).
Reader  injected at **layer 2** -- the PLE layer per ``docs/real-weights-spec.json``
        (``ple_layer_ids=[2]``).
Table   the real Qwen3.8-Flash-Next PLE shards (128 x 400,001,920 B, width 160).

Arms
----
none      no reader                      -> engine ceiling
shm       rows preloaded into /dev/shm   -> "the table is free" world
mmap      np.memmap over the NVMe shards -> what engines do natively
engram-i  engramdb.Store.fetch           -> Store-I, N scattered badge reads

Honest boundaries (read before quoting any number)
--------------------------------------------------
* ``enforce_eager=True``.  A Python-level reader cannot be captured into a CUDA
  graph, so this measures the **eager** serving path.  Every arm pays the same
  penalty, so the A/B is fair; the absolute tok/s is NOT a CUDA-graph number.
* The table is 25 GiB and this host has 1 TB of RAM, so the page cache will
  swallow it.  "Cold" is therefore **manufactured** with
  ``posix_fadvise(POSIX_FADV_DONTNEED)`` and must pass a self-check, exactly as
  the rest of this repo does.  Without the check the numbers are void.
* The reader output is projected into the residual stream with a fixed random
  matrix.  This measures the **storage cost** of a PLE-shaped lookup; it does
  NOT make the model a trained PLE model, and no quality claim follows.

Usage
-----
    python serve_ple_ab.py --arm none    --json-out /tmp/ab-none.json
    python serve_ple_ab.py --arm engram-i --rows-per-token 16 --iterations 3
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
import time
from pathlib import Path

MODEL = os.environ.get("ENGRAMDB_SERVE_MODEL", "/root/autodl-tmp/qwen35-ple/models/Qwen3.5-0.8B")
ROWS_DIR = os.environ.get("ENGRAMDB_PLE_ROWS", "/root/autodl-tmp/qwen35-ple/qwen38-rows")

SHARD_BYTES = 400_001_920
ROW_WIDTH = 160
ROWS_PER_SHARD = SHARD_BYTES // ROW_WIDTH  # 2,500,012


# --------------------------------------------------------------------------- #
# cold-cache discipline
# --------------------------------------------------------------------------- #
def _fadvise_dontneed(path: str) -> bool:
    """Drop this file's pages from the page cache.

    Linux-only: ``POSIX_FADV_DONTNEED`` does not exist on macOS/BSD libc.
    Returns False when the call is unavailable so the caller can mark the run
    unverifiable instead of silently reporting warm numbers as cold.
    """
    if not sys.platform.startswith("linux"):
        return False
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    POSIX_FADV_DONTNEED = 4
    fd = os.open(path, os.O_RDONLY)
    try:
        rc = libc.posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED)
        return rc == 0
    finally:
        os.close(fd)


def drop_table_cache(shards: list[Path]) -> int:
    return sum(1 for p in shards if _fadvise_dontneed(str(p)))


#: Minimum cold/warm ratio for a run to be allowed to call itself "cold".
COLD_RATIO_MIN = 5.0
#: ...or a minimum per-read marginal over warm, in microseconds.
COLD_MARGINAL_US_MIN = 2.0


def verify_cold(shards: list[Path], sample: int = 8) -> dict:
    """Prove that ``drop_table_cache`` actually evicted something.

    Without this the word "cold" is a *claim*, not a measurement.  The
    2026-09-12 run is the cautionary case: after ``fadvise(DONTNEED)`` on all
    128 shards of a 48 GB table, the reader cost moved only 185 -> 230 us, which
    is what a *failed* eviction looks like -- yet the log said "cold".

    Method: pick ``sample`` random 4 KiB pages, read them once (cold), read the
    same pages again (warm), compare.  The caller must have already called
    ``drop_table_cache``.

    Returns a dict with ``ok`` False meaning the run must be reported as VOID
    rather than as a cold number.
    """
    import random

    fds = [os.open(p, os.O_RDONLY) for p in shards]
    try:
        rng = random.Random(0xC01D)
        picks = [
            (rng.randrange(len(fds)), rng.randrange(0, SHARD_BYTES - 4096, 4096))
            for _ in range(sample)
        ]
        t0 = time.perf_counter()
        for i, off in picks:
            os.pread(fds[i], 4096, off)
        cold = time.perf_counter() - t0
        t0 = time.perf_counter()
        for i, off in picks:
            os.pread(fds[i], 4096, off)
        warm = time.perf_counter() - t0
    finally:
        for fd in fds:
            os.close(fd)

    ratio = (cold / warm) if warm > 0 else float("inf")
    marginal_us = (cold - warm) / sample * 1e6
    ok = ratio >= COLD_RATIO_MIN or marginal_us >= COLD_MARGINAL_US_MIN
    return {
        "ok": ok,
        "cold_us_total": cold * 1e6,
        "warm_us_total": warm * 1e6,
        "ratio": ratio,
        "marginal_us_per_read": marginal_us,
        "sample": sample,
        "verdict": "cold" if ok else "VOID (eviction did not take effect)",
    }


# --------------------------------------------------------------------------- #
# readers
# --------------------------------------------------------------------------- #
class BaseReader:
    name = "base"
    #: bytes actually demanded from the medium per token (payload, not pages)
    payload_bytes_per_token = 0
    #: 4 KiB pages touched per token (what the medium actually transfers)
    pages_per_token = 0

    def fetch(self, token_ids) -> "object":  # -> torch.Tensor [n, out_dim] cpu
        raise NotImplementedError

    def stats(self) -> dict:
        return {}


def _flatten_rowids(rowids: list[list[int]]) -> list[int]:
    return [r for row in rowids for r in row]


class EngramStoreReader(BaseReader):
    """Store-I: EngramDB's own fetch over the real shard directory."""

    name = "engram-i"

    def __init__(self, rows_dir: str, rows_per_token: int, row_width: int = ROW_WIDTH):
        import engramdb

        n_shards = len(list(Path(rows_dir).glob("shard_*.bin")))
        self._shards = sorted(Path(rows_dir).glob("shard_*.bin"))
        self._n_shards = n_shards
        self._rows_total = n_shards * (SHARD_BYTES // row_width)
        self.store = engramdb.Store(rows_dir, n_shards, SHARD_BYTES // row_width, row_width)
        self._engramdb = engramdb
        self.rows_per_token = rows_per_token
        self.row_width = row_width
        self.payload_bytes_per_token = rows_per_token * row_width
        self.pages_per_token = rows_per_token  # one 4 KiB page fault per row
        self._fetch_s = 0.0
        self._calls = 0

    def _rowids(self, token_ids) -> list[list[int]]:
        """Real EngramDB PLE rowids (``PLE_QWEN_V1``), 16 heads per token.

        Row ids come from EngramDB's own keygen over the full padded table
        space (320,001,536 rows = 128 shards x 2,500,012).  The modulo below
        exists only for the case where the local table is a partial download
        (the first serving run had 65 of 128 shards): it folds ids into the rows
        that actually exist.

        With all 128 shards present the modulo is a **no-op**: keygen's largest
        possible id is ``total_vocab - 1 = 320,001,445``, which is below
        ``_rows_total = 320,001,536``.  So a full-table run needs no special
        flag -- just have the shards there.  Verified by
        ``probes/ple_rowid_exactness_session42.md`` (padded_vocab == on-disk
        rows == 320,001,536).
        """
        toks = [int(t) for t in token_ids]
        ids = self._engramdb.rowids_for_seq(toks, self._engramdb.PLE_QWEN_V1)
        out = []
        for row in ids:
            row = [int(r) % self._rows_total for r in row]
            if len(row) < self.rows_per_token:  # probe V4.1-style row counts
                row = (row * (self.rows_per_token // len(row) + 1))[: self.rows_per_token]
            out.append(row[: self.rows_per_token])
        return out

    def fetch(self, token_ids):
        import torch

        t0 = time.perf_counter()
        rowids = self._rowids(token_ids)
        flat = _flatten_rowids(rowids)
        raw = self.store.fetch(flat)
        self._fetch_s += time.perf_counter() - t0
        self._calls += 1
        return torch.frombuffer(bytearray(raw), dtype=torch.uint8).view(torch.float32).reshape(
            len(token_ids), -1
        )

    def stats(self) -> dict:
        return {
            "reader_fetch_s_total": self._fetch_s,
            "reader_calls": self._calls,
            "reader_us_per_call": (self._fetch_s / self._calls * 1e6) if self._calls else None,
        }


class MmapReader(BaseReader):
    """Native-engine baseline: mmap the shards and gather with numpy."""

    name = "mmap"

    def __init__(self, rows_dir: str, rows_per_token: int):
        import numpy as np

        self._np = np
        self._shards = sorted(Path(rows_dir).glob("shard_*.bin"))
        self.maps = [np.memmap(p, dtype=np.uint8, mode="r") for p in self._shards]
        self._n_shards = len(self.maps)
        self._rows_total = self._n_shards * ROWS_PER_SHARD
        self.rows_per_token = rows_per_token
        self.payload_bytes_per_token = rows_per_token * ROW_WIDTH
        self.pages_per_token = rows_per_token
        self._fetch_s = 0.0
        self._calls = 0

    def _rowids(self, n: int) -> list[int]:
        # Deterministic spread over the whole table.  The mmap arm is a
        # *medium* baseline; rowid provenance is not its variable.
        import random

        rng = random.Random(0xE11)
        return [rng.randrange(self._rows_total) for _ in range(n * self.rows_per_token)]

    def fetch(self, token_ids):
        import torch

        t0 = time.perf_counter()
        n = len(token_ids)
        rids = self._rowids(n)
        buf = bytearray(n * self.payload_bytes_per_token)
        for i, rid in enumerate(rids):
            sh = rid // ROWS_PER_SHARD
            off = (rid % ROWS_PER_SHARD) * ROW_WIDTH
            # numpy slice -> bytearray needs an explicit copy to bytes
            buf[i * ROW_WIDTH : (i + 1) * ROW_WIDTH] = self.maps[sh][
                off : off + ROW_WIDTH
            ].tobytes()
        self._fetch_s += time.perf_counter() - t0
        self._calls += 1
        return torch.frombuffer(buf, dtype=torch.uint8).view(torch.float32).reshape(n, -1)

    def stats(self) -> dict:
        return {
            "reader_fetch_s_total": self._fetch_s,
            "reader_calls": self._calls,
            "reader_us_per_call": (self._fetch_s / self._calls * 1e6) if self._calls else None,
        }

    def close(self) -> None:
        for m in getattr(self, "maps", []):
            del m
        self.maps = []


class ShmReader(BaseReader):
    """Upper bound: the same bytes, served from /dev/shm."""

    name = "shm"

    def __init__(self, rows_dir: str, rows_per_token: int, budget_bytes: int):
        import shutil

        self._shards = sorted(Path(rows_dir).glob("shard_*.bin"))
        self.dir = Path("/dev/shm/engram-ple-rows")
        self.dir.mkdir(parents=True, exist_ok=True)
        used = 0
        self._copied = []
        for p in self._shards:
            if used + p.stat().st_size > budget_bytes:
                break
            dst = self.dir / p.name
            if not dst.exists():
                shutil.copyfile(p, dst)
            used += p.stat().st_size
            self._copied.append(dst)
        self._fds = [os.open(p, os.O_RDONLY) for p in self._copied]
        self._rows_total = len(self._copied) * ROWS_PER_SHARD
        self.rows_per_token = rows_per_token
        self.payload_bytes_per_token = rows_per_token * ROW_WIDTH
        self.pages_per_token = rows_per_token
        self._fetch_s = 0.0
        self._rowid_s = 0.0
        self._calls = 0

    def _rowids_for(self, token_ids):
        """Fixed-RNG spread over the whole table: no keygen cost at all.

        Overridden by ``ShmKeygenReader`` to use the real EngramDB keygen, so
        that keygen cost and storage cost can be measured apart.
        """
        import random

        rng = random.Random(0xE11)
        return [
            [rng.randrange(self._rows_total) for _ in range(self.rows_per_token)]
            for _ in token_ids
        ]

    def fetch(self, token_ids):
        import torch

        t0 = time.perf_counter()
        rowids = self._rowids_for(token_ids)
        t1 = time.perf_counter()
        buf = bytearray(self.payload_bytes_per_token * len(rowids))
        i = 0
        for row in rowids:
            for rid in row:
                sh = rid // ROWS_PER_SHARD
                off = (rid % ROWS_PER_SHARD) * ROW_WIDTH
                buf[i * ROW_WIDTH : (i + 1) * ROW_WIDTH] = os.pread(
                    self._fds[sh], ROW_WIDTH, off
                )
                i += 1
        self._rowid_s += t1 - t0
        self._fetch_s += time.perf_counter() - t0
        self._calls += 1
        return torch.frombuffer(buf, dtype=torch.uint8).view(torch.float32).reshape(
            len(rowids), -1
        )

    def stats(self) -> dict:
        return {
            "shm_shards": len(self._copied),
            "shm_bytes": sum(p.stat().st_size for p in self._copied),
            "reader_fetch_s_total": self._fetch_s,
            "rowid_s_total": self._rowid_s,
            "reader_calls": self._calls,
            "reader_us_per_call": (self._fetch_s / self._calls * 1e6) if self._calls else None,
            "rowid_us_per_call": (self._rowid_s / self._calls * 1e6) if self._calls else None,
        }


class ShmKeygenReader(ShmReader):
    """Real EngramDB keygen + /dev/shm I/O.

    Pairing this with ``engram-i`` and plain ``shm`` separates the two costs
    that the storage A/B otherwise conflates:

        shm          fixed RNG   + RAM   -> the I/O+Python path, no keygen
        shm-keygen   real keygen + RAM   -> keygen cost, I/O free
        engram-i     real keygen + disk  -> both

    ``(engram-i) - (shm-keygen)`` is the storage cost with keygen netted out.
    Without this arm the whole drop gets attributed to the disk, which is what
    the first version of probes/serve_ple_ab_session42.md did -- wrongly.
    """

    name = "shm-keygen"

    def _rowids_for(self, token_ids):
        import engramdb

        toks = [int(t) for t in token_ids]
        ids = engramdb.rowids_for_seq(toks, engramdb.PLE_QWEN_V1)
        out = []
        for row in ids:
            r = [int(x) % self._rows_total for x in row]
            if len(r) < self.rows_per_token:
                r = (r * (self.rows_per_token // len(r) + 1))[: self.rows_per_token]
            out.append(r[: self.rows_per_token])
        return out


class PrefetchUpperBoundReader(BaseReader):
    """TIMING-ONLY probe: a real fetch, consumed one step late.

    The values returned are the *previous* step's rows, so the model's output
    is deliberately wrong.  What this measures is the timing of a prefetch
    issued when the token is sampled: the fetch then has a whole inter-step gap
    (scheduler, Python, H2D, prefill of the next step) to finish, so it should
    never block layer L.

    It is the **upper bound** on what prefetching can buy.  Never quote its
    output as a result, and never mix its tok/s into a table without the label.
    """

    name = "prefetch-ub"

    def __init__(self, inner: BaseReader):
        import threading

        self.inner = inner
        self.rows_per_token = inner.rows_per_token
        self.payload_bytes_per_token = inner.payload_bytes_per_token
        self.pages_per_token = inner.pages_per_token
        self._last = None
        self._lock = threading.Lock()
        self._thread = None
        self._fetch_s = 0.0
        self._calls = 0

    def fetch(self, token_ids):
        import threading

        import torch

        n = len(token_ids)
        if n > 4:
            # Prefill: a real design would fetch the whole batch before the
            # layer anyway, so this stays synchronous and its cost is honest.
            return self.inner.fetch(token_ids)

        with self._lock:
            prev, self._last = self._last, None
            if self._thread is not None:
                self._thread.join()

        def work():
            t0 = time.perf_counter()
            out = self.inner.fetch(token_ids)
            self._fetch_s += time.perf_counter() - t0
            with self._lock:
                self._last = out

        self._calls += 1
        self._thread = threading.Thread(target=work, daemon=True)
        self._thread.start()

        # The *values* are stale on purpose; only the timing is meaningful.
        # Shapes must still match the current step or the residual add breaks.
        if prev is None or prev.shape[0] != n:
            return torch.zeros(n, self.payload_bytes_per_token // 4, dtype=torch.float32)
        return prev

    def stats(self) -> dict:
        st = dict(self.inner.stats())
        st.update(
            {
                "reader_calls": self._calls,
                "prefetch_upper_bound": True,
                "values_are_stale": True,
            }
        )
        return st

    def close(self) -> None:
        if self._thread is not None:
            self._thread.join()
        close = getattr(self.inner, "close", None)
        if callable(close):
            close()


# --------------------------------------------------------------------------- #
# injection
# --------------------------------------------------------------------------- #
STATE: dict = {}


def find_layer_list(model):
    """Locate the transformer layer ModuleList by size, not by hard-coded path."""
    from torch import nn

    best = (None, None)
    for name, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) >= 8:
            if best[1] is None or len(mod) > len(best[1]):
                best = (name, mod)
    return best


def find_model(llm):
    """Locate the ``nn.Module`` that holds the transformer layers.

    vLLM V1 normally runs the engine core in a **separate process**, so a hook
    installed from the parent process would never touch the model.  The caller
    must therefore set ``VLLM_ENABLE_V1_MULTIPROCESSING=0`` first; this function
    then walks the known handle chain instead of hard-coding one attribute
    path, because that path has moved between vLLM versions.
    """
    from torch import nn

    attrs = (
        "llm_engine",
        "engine_core",
        "engine_core_client",
        "model_executor",
        "driver_worker",
        "model_runner",
        "worker",
        "model",
    )
    seen: set = set()
    queue = [llm]
    while queue:
        obj = queue.pop(0)
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        if isinstance(obj, nn.Module) and find_layer_list(obj)[1] is not None:
            return obj
        for name in attrs:
            try:
                val = getattr(obj, name, None)
            except Exception:
                continue
            if val is not None and id(val) not in seen:
                queue.append(val)
    return None


def install_reader(llm, holder: dict, layer_idx: int, out_dim: int):
    """Wrap the top model to stash input_ids, then hook layer ``layer_idx``.

    ``holder`` is mutable so the reader can be swapped between arms without
    rebuilding the engine (a rebuild costs ~2 min of model load per arm).
    """
    import types

    import torch

    runner = None
    model = find_model(llm)
    if model is None:
        raise RuntimeError(
            "could not find a model with a transformer layer list. "
            "Did you set VLLM_ENABLE_V1_MULTIPROCESSING=0? Without it the model "
            "lives in another process and no in-process hook can reach it."
        )
    lname, layers = find_layer_list(model)
    if layers is None or layer_idx >= len(layers):
        raise RuntimeError(f"no layer list found (got {lname!r}, len={len(layers) if layers else 0})")

    hidden = None
    emb = None
    emb_name = None
    for name, mod in model.named_modules():
        if name.endswith("embed_tokens"):
            hidden = mod.weight.shape[1]
            emb, emb_name = mod, name
            break
    if hidden is None:
        raise RuntimeError("no embed_tokens found; cannot infer hidden size")

    proj = torch.randn(out_dim, hidden, device="cuda", dtype=torch.float32) * (hidden ** -0.5)

    # 1. Two INDEPENDENT stash points for input_ids.  vLLM's call path has
    #    moved between versions: patching the top-level forward is not always
    #    enough, and a hook that never fires produces a beautiful, completely
    #    false "zero overhead" result.  So we install both and count each.
    target = model
    orig = target.forward

    def wrapper(self, *a, **kw):
        ids = kw.get("input_ids") if kw else None
        if ids is None and a:
            ids = a[0]
        if ids is not None:
            STATE["input_ids"] = ids
            STATE["stash_forward"] = STATE.get("stash_forward", 0) + 1
        return orig(*a, **kw)

    target.forward = types.MethodType(wrapper, target)

    def emb_pre(module, args, kwargs=None):
        ids = None
        if kwargs:
            ids = kwargs.get("input_ids")
        if ids is None and args:
            ids = args[0]
        if ids is not None:
            STATE["input_ids"] = ids
            STATE["stash_embed"] = STATE.get("stash_embed", 0) + 1

    emb.register_forward_pre_hook(emb_pre, with_kwargs=True)

    def hook(module, args, output):
        reader = holder.get("reader")
        ids = STATE.get("input_ids")
        STATE["layer_hits"] = STATE.get("layer_hits", 0) + 1
        if reader is None or ids is None:
            return output
        hs = output[0] if isinstance(output, tuple) else output
        n = hs.shape[0]
        tok = ids.reshape(-1)[-n:]
        e = reader.fetch([int(t) for t in tok])
        # proj is (out_dim, hidden): (n, out_dim) @ (out_dim, hidden) -> (n, hidden)
        e = e.to(device=hs.device, dtype=torch.float32) @ proj
        hs = hs + e.to(hs.dtype)
        return (hs,) + tuple(output[1:]) if isinstance(output, tuple) else hs

    h = layers[layer_idx].register_forward_hook(hook)
    cfg = getattr(model, "config", None)
    n_cfg = None
    for src in (cfg, getattr(cfg, "text_config", None)):
        if src is not None and getattr(src, "num_hidden_layers", None):
            n_cfg = src.num_hidden_layers
            break
    if n_cfg is not None and n_cfg != len(layers):
        print(
            f"[inject] WARNING: picked layer list {lname!r} has {len(layers)} entries "
            f"but config says num_hidden_layers={n_cfg}; the hook may be on the wrong stack"
        )
    print(
        f"[inject] layers={lname} n={len(layers)} config_layers={n_cfg} "
        f"layer={layer_idx} hidden={hidden} out_dim={out_dim} "
        f"proj={tuple(proj.shape)}"
    )
    return h, hidden


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--arms",
        default="none",
        help="comma list of none,shm,mmap,engram-i.  All arms run in ONE process: "
        "rebuilding the engine per arm costs ~2 min of model load each, and it "
        "would also let the two runs differ in engine state.",
    )
    ap.add_argument("--rows-per-token", type=int, default=16)
    ap.add_argument("--layer", type=int, default=2)
    ap.add_argument("--iterations", type=int, default=3)
    ap.add_argument("--prompt-len", type=int, default=128)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--shm-budget-gb", type=float, default=20.0)
    ap.add_argument("--cold", action="store_true", help="fadvise DONTNEED before each iteration")
    ap.add_argument(
        "--selfcheck-cache",
        action="store_true",
        help="only verify that fadvise actually evicts, then exit; needs no GPU/vLLM",
    )
    ap.add_argument("--gpu-mem-util", type=float, default=0.85)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    # The reader is a Python hook, so the model must live in THIS process.
    # vLLM V1 otherwise runs the engine core in a child process; the hook would
    # then silently never fire and every arm would look identically fast --
    # a beautiful, completely false "no overhead" result.
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    # Cache self-check runs BEFORE importing vllm: it needs no engine and no
    # GPU, so it can be used to answer "is fadvise even working on this box?"
    # without paying a model load.
    if args.selfcheck_cache:
        sc_shards = sorted(Path(ROWS_DIR).glob("shard_*.bin"))
        if not sc_shards:
            print(f"FATAL: no shards under {ROWS_DIR}", file=sys.stderr)
            return 2
        n = drop_table_cache(sc_shards)
        v = verify_cold(sc_shards)
        print(
            f"[selfcheck] fadvise on {n}/{len(sc_shards)} shards "
            f"({sum(p.stat().st_size for p in sc_shards) / 2**30:.1f} GiB)\n"
            f"[selfcheck] cold={v['cold_us_total']:.1f}us warm={v['warm_us_total']:.1f}us "
            f"ratio={v['ratio']:.1f}x marginal={v['marginal_us_per_read']:.1f}us/read\n"
            f"[selfcheck] verdict: {v['verdict']}"
        )
        if args.json_out:
            Path(args.json_out).write_text(json.dumps(v, indent=2) + "\n")
        return 0 if v["ok"] else 3

    from vllm import LLM, SamplingParams

    shards = sorted(Path(ROWS_DIR).glob("shard_*.bin"))
    print(f"[table] {len(shards)} shards in {ROWS_DIR}")
    if not shards:
        print("FATAL: no shards", file=sys.stderr)
        return 2

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    holder: dict = {"reader": None}

    llm = LLM(
        model=MODEL,
        enforce_eager=True,          # a Python reader cannot be CUDA-graphed
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=4096,
        disable_log_stats=True,
        trust_remote_code=True,
    )

    out_dim = args.rows_per_token * ROW_WIDTH // 4
    handle = None
    if any(a != "none" for a in arms):
        handle, hidden = install_reader(llm, holder, args.layer, out_dim)
        print(f"[inject] ok, hidden={hidden} out_dim={out_dim}")

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    # Feed token ids directly: decoding arbitrary ids to text and re-encoding
    # does not round-trip, so the rowids would not match the ids we intended.
    prompts = [
        {
            "prompt_token_ids": [
                1000 + (7 * i + j) % 200000 for j in range(args.prompt_len)
            ]
        }
        for i in range(args.batch)
    ]
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, ignore_eos=True)

    def make_reader(arm: str):
        if arm == "engram-i":
            return EngramStoreReader(ROWS_DIR, args.rows_per_token)
        if arm == "mmap":
            return MmapReader(ROWS_DIR, args.rows_per_token)
        if arm == "shm":
            return ShmReader(ROWS_DIR, args.rows_per_token, int(args.shm_budget_gb * 2**30))
        if arm == "shm-keygen":
            return ShmKeygenReader(
                ROWS_DIR, args.rows_per_token, int(args.shm_budget_gb * 2**30)
            )
        if arm == "prefetch-ub":
            return PrefetchUpperBoundReader(
                EngramStoreReader(ROWS_DIR, args.rows_per_token)
            )
        if arm == "prefetch-ub-shm":
            return PrefetchUpperBoundReader(
                ShmReader(ROWS_DIR, args.rows_per_token, int(args.shm_budget_gb * 2**30))
            )
        return None

    results = []
    cold_checks: list = []
    sample_text = None
    for arm in arms:
        reader = make_reader(arm)
        holder["reader"] = reader
        for k in ("stash_forward", "stash_embed", "layer_hits"):
            STATE[k] = 0
        runs = []
        for it in range(args.iterations):
            if args.cold and arm not in ("none", "shm"):
                n = drop_table_cache(shards)
                v = verify_cold(shards)
                cold_checks.append(v)
                print(
                    f"[cold] {arm} iter={it}: fadvise on {n}/{len(shards)} shards; "
                    f"ratio={v['ratio']:.1f}x marginal={v['marginal_us_per_read']:.1f}us "
                    f"-> {v['verdict']}"
                )
                if not v["ok"]:
                    print(
                        "  >>> this iteration's numbers are VOID: the cache was not "
                        "actually evicted, so 'cold' would be a false claim"
                    )
            t0 = time.perf_counter()
            outs = llm.generate(prompts, sp)
            dt = time.perf_counter() - t0
            ntok = sum(len(o.outputs[0].token_ids) for o in outs)
            runs.append({"iter": it, "seconds": dt, "tokens": ntok, "tok_per_s": ntok / dt})
            print(
                f"[run] arm={arm} iter={it} tokens={ntok} s={dt:.3f} tok/s={ntok/dt:.1f}"
                + (f" reader_us/call={reader.stats().get('reader_us_per_call'):.1f}"
                   if reader and reader.stats().get("reader_us_per_call") else "")
            )
        if sample_text is None:
            sample_text = outs[0].outputs[0].text[:200]
        vals = sorted(r["tok_per_s"] for r in runs)
        results.append(
            {
                "arm": arm,
                "runs": runs,
                "tok_per_s_median": vals[len(vals) // 2],
                "tok_per_s_best": vals[-1],
                "reader": reader.stats() if reader else {},
                "payload_bytes_per_token": (
                    reader.payload_bytes_per_token if reader else 0
                ),
                "pages_per_token": reader.pages_per_token if reader else 0,
                # Instrumentation: if all three are 0 the hook is dead code and
                # every number in this arm is meaningless.
                "diag": {
                    "stash_forward": STATE.get("stash_forward", 0),
                    "stash_embed": STATE.get("stash_embed", 0),
                    "layer_hits": STATE.get("layer_hits", 0),
                },
            }
        )
        if reader is not None:
            close = getattr(reader, "close", None)
            if callable(close):
                close()

    baseline = next((r for r in results if r["arm"] == "none"), None)
    for r in results:
        if baseline and baseline["tok_per_s_median"]:
            r["tok_per_s_vs_none_pct"] = (
                100.0 * (r["tok_per_s_median"] - baseline["tok_per_s_median"])
                / baseline["tok_per_s_median"]
            )
            r["added_us_per_token"] = (
                1e6 / r["tok_per_s_median"] - 1e6 / baseline["tok_per_s_median"]
            )

    result = {
        "engine": "vllm",
        "vllm": __import__("vllm").__version__,
        "enforce_eager": True,
        "model": MODEL,
        "layer": args.layer,
        "rows_per_token": args.rows_per_token,
        "row_width": ROW_WIDTH,
        "batch": args.batch,
        "prompt_len": args.prompt_len,
        "max_tokens": args.max_tokens,
        "cold": args.cold,
        "table_shards": len(shards),
        "table_gib": sum(p.stat().st_size for p in shards) / 2**30,
        "cold_checks": cold_checks,
        "cold_verified": (all(c["ok"] for c in cold_checks) if cold_checks else None),
        "results": results,
        "sample_output": sample_text,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n"
        )
    if handle is not None:
        handle.remove()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
