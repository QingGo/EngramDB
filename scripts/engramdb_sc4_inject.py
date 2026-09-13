#!/usr/bin/env python3
"""Sub-condition 4, SGLang side: a disk read that executes *inside* graph replay.

Why this file exists
--------------------
vLLM 0.29.0 could not be made to do this.  Eight runs are archived in
``probes/sc4_*.json``; the eager arms all passed and every graph arm was VOID,
because vLLM's dispatcher prefers ``FULL`` over ``PIECEWISE`` for decode, so the
``splitting_ops`` boundary that would have hosted the read is bypassed and no
Python runs at replay at all (roadmap §35.1d).

SGLang's **breakable CUDA graph** is a different mechanism, not a different
configuration.  ``eager_on_graph`` ends the current stream-capture segment,
runs the wrapped body once so its outputs get allocated, and records a
``replay_fn``.  ``BreakableCUDAGraph.replay`` then does::

    for i, seg in enumerate(self._segments):
        seg.replay()                    # async cudaGraphLaunch
        if i < len(self._break_fns):
            self._break_fns[i]()        # <-- real host Python, no sync

So host code -- and therefore disk I/O -- runs *between* the captured segments,
with no synchronisation forced.  That is the whole mechanism this file tests.

Three arms
----------
``none``   no injection at all: the engine floor in breakable-graph mode.
``break``  the break is installed and runs, but its delta is synthesised from the
           token ids with **no disk I/O**.  Isolates the cost of the break
           mechanism itself (host call + H2D + ``add_``), and doubles as the
           negative control for the token-difference test.
``read``   the break performs the real 16-row read from the on-disk table, with
           the rowids computed from the live token ids.
``read_static``
           the same disk read, but the rowids are **precomputed host-side** and
           the break does **no D2H at all**.  ``read - read_static`` is therefore
           the cost of the device-to-host sync plus the rowid computation, which
           is the term that has to disappear before ``read - break`` can be
           called a storage cost.

``read - break`` is therefore the cost of the disk read **inside a graph step**,
and ``break - none`` is the cost of the mechanism that makes it possible.  Both
are measured under the same regime; neither is inferred from an eager run.

How the arms stay comparable
----------------------------
The break body is the *same callable* in both injected arms -- only its data
source differs -- and outside a capture ``eager_on_graph`` runs the body inline,
so the eager and graph paths execute identical code.

Self-proof (this is the part that has to be airtight)
----------------------------------------------------
No counter can be placed inside a captured graph, so the proof is layered:

1. ``BreakableCudaGraphBackend.replay`` is wrapped and counted.  If this is zero
   the graph was never replayed and every other number is an eager number --
   this is exactly the ``PIECEWISE -> NONE`` silent degradation that makes a
   false pass possible, so it is checked first.
2. ``BreakableCUDAGraph.replay`` sets a flag for the duration of its loop, and
   the break body records how many times it ran **while that flag was set**.
   This is the direct evidence for the acceptance criterion.  Note that
   ``is_in_breakable_cuda_graph()`` is deliberately *not* used as the
   discriminator: ``replay_session`` wraps replay, but the same context manager
   is entered during capture, so it is True in both phases and would pass
   vacuously.
3. Functional: the ``read``/``break`` arms must produce **different greedy
   tokens** than ``none``.  The add happens outside the captured segments, so if
   the body did not run at replay the delta would never reach the model and the
   tokens would match the baseline exactly.
4. Freshness: the token ids the body reads at replay are recorded, so staleness
   (reading the capture-time batch instead of the live one) is visible rather
   than assumed away.

Usage
-----
Not run directly.  ``scripts/qwen3_5_sc4_hook.py`` appends a loader to the
installed ``sglang/srt/models/qwen3_5.py``; this module then self-installs from
the environment::

    ENGRAMDB_SC4_ARM=read
    ENGRAMDB_SC4_LAYER=1
    ENGRAMDB_SC4_ROWS=16
    ENGRAMDB_SC4_STORE=/root/autodl-tmp/qwen35-ple/qwen38-rows
    ENGRAMDB_SC4_COUNTERS=/tmp/sc4_counters.json
"""

from __future__ import annotations

import atexit
import json
import os
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor

# Real geometry of Qwen/Qwen3.8-Flash-Next-FP8's PLE table, verified against the
# checkpoint's own safetensors index (see docs/engram-specs.md):
# 128 tensors x 2_500_012 rows x 160 B (F8_E4M3) = 320_001_536 rows = 51.2 GB.
ROWS_PER_SHARD = 2_500_012
ROW_BYTES = 160
N_SHARDS = 128

_STATE: dict = {
    "arm": "none",
    "layer": 1,
    "rows": 16,
    "store": None,
    "counters": None,
}
_C: dict = {
    "break_calls": 0,
    "break_calls_in_replay": 0,
    "break_calls_in_capture": 0,
    "forward_calls": 0,
    "forward_by_class": {},
    "backend_replay_calls": 0,
    "bcg_replay_calls": 0,
    "rows_read": 0,
    "bytes_read": 0,
    "read_us": [],
    "break_us": [],
    "tok_samples": [],
    "errors": [],
    "skipped": [],
    "wrapped_classes": [],
}
_KEEP = 4000          # bound the in-memory sample lists
_lock = threading.Lock()
_INSTALLED = False
_IN_REPLAY = False
_FDS: list[int] = []
_POOL: ThreadPoolExecutor | None = None
_HOSTBUF = None


# ---------------------------------------------------------------------------
# the on-disk table
# ---------------------------------------------------------------------------

def _open_store(store: str) -> None:
    global _FDS, _POOL, _HOSTBUF
    import numpy as np

    if _FDS:
        return
    for i in range(N_SHARDS):
        p = os.path.join(store, f"shard_{i:03d}.bin")
        _FDS.append(os.open(p, os.O_RDONLY))
    # One thread per outstanding read; the table's cold cost is page-fault
    # dominated (one 4 KiB page per 160 B row), so concurrency is what makes it
    # schedulable at all -- see scripts/lead_layer_budget.py.
    _POOL = ThreadPoolExecutor(max_workers=min(64, N_SHARDS))
    _HOSTBUF = np.zeros(0, dtype=np.uint8)


def _rowid_to_loc(rowid: int) -> tuple[int, int]:
    return divmod(int(rowid), ROWS_PER_SHARD)


def _read_rows(rowids, out: "np.ndarray") -> None:
    """Fill ``out`` (uint8 [n, rows*ROW_BYTES]) with the rows named by ``rowids``.

    ``os.pread`` on a plain fd rather than an mmap: an mmap keeps pages resident
    through the mapping, which makes ``posix_fadvise(DONTNEED)`` unable to evict
    them and silently turns a cold measurement warm (the mistake recorded in
    ``probes/serve_ple_ab_fulltable_session42.md``).
    """
    import numpy as np

    n, width = out.shape
    flat = out.reshape(-1)

    def one(job):
        idx, rowid = job
        shard, off = _rowid_to_loc(rowid)
        data = os.pread(_FDS[shard], ROW_BYTES, off * ROW_BYTES)
        if len(data) != ROW_BYTES:
            raise IOError(f"short read: rowid={rowid} got {len(data)}")
        return idx, data

    jobs = ((i * _STATE["rows"] + r, int(rowids[i, r]))
            for i in range(n) for r in range(_STATE["rows"]))
    # chunksize=1, deliberately.  With 16 jobs and chunksize=8, map() hands out
    # two chunks, so only **two** workers run and the 16 page faults serialise
    # two at a time.  That is the whole gap between this probe's ~461 us per
    # token and the 195.9 us measured with real concurrency: 16 x 85.8 us / 2
    # ~= 686 us observed versus 16 x 85.8 / 16 ~= 86 us ideal, and 12.2 us/row
    # is what the tuned run actually achieved.
    for idx, data in _POOL.map(one, jobs, chunksize=1):
        s = idx * ROW_BYTES
        flat[s:s + ROW_BYTES] = np.frombuffer(data, dtype=np.uint8)
    assert width == _STATE["rows"] * ROW_BYTES


def drop_page_cache(store: str) -> None:
    """Best-effort cold: evict the table's pages so a 'cold' label is earned."""
    for i in range(N_SHARDS):
        fd = os.open(os.path.join(store, f"shard_{i:03d}.bin"), os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)


# ---------------------------------------------------------------------------
# the break body
# ---------------------------------------------------------------------------

def _delta_from_tokens(toks, rows, host: "np.ndarray"):
    """arm='break': a deterministic, token-dependent delta with no disk I/O."""
    import numpy as np

    n = len(toks)
    w = min(1024, host.shape[1])
    for i, t in enumerate(toks):
        seed = (int(t) * 2654435761) & 0xFFFFFFFF
        block = np.arange(w, dtype=np.uint32) * 2246822519 + seed
        host[i, :w] = ((block >> 16) & 0xFF).astype(np.uint8)
    return n


def _delta_from_disk(toks, rowids, host: "np.ndarray"):
    """arm='read': the real 16-row read for every token in the batch."""
    import numpy as np

    n = len(toks)
    raw = np.zeros((n, _STATE["rows"] * ROW_BYTES), dtype=np.uint8)
    t0 = time.perf_counter()
    _read_rows(rowids, raw)
    dt = time.perf_counter() - t0
    with _lock:
        _C["rows_read"] += n * _STATE["rows"]
        _C["bytes_read"] += int(raw.nbytes)
        if len(_C["read_us"]) < _KEEP:
            _C["read_us"].append(dt * 1e6)
    # Project the 16x160 B (fp8) rows onto the model's hidden width.  The *cost*
    # is the read; the projection is synthetic and claims nothing about quality.
    flat = raw.reshape(n, -1)
    h = host.shape[1]
    reps = (h + flat.shape[1] - 1) // flat.shape[1]
    host[:, :] = np.tile(flat, (1, reps))[:, :h]
    return n


def install() -> bool:
    """Install the break into the Qwen3.5 decoder layers.  Idempotent."""
    global _INSTALLED, _HOSTBUF
    if _INSTALLED:
        return True
    arm = os.environ.get("ENGRAMDB_SC4_ARM", "")
    if not arm or arm == "none":
        return False

    import numpy as np
    import torch

    from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
        eager_on_graph,
    )
    from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.breakable_cuda_graph import (
        BreakableCUDAGraph,
    )
    from sglang.srt.model_executor.runner_backend.breakable_cuda_graph_backend import (
        BreakableCudaGraphBackend,
    )
    from sglang.srt.models import qwen3_5

    _STATE["arm"] = arm
    _STATE["layer"] = int(os.environ.get("ENGRAMDB_SC4_LAYER", "1"))
    _STATE["rows"] = int(os.environ.get("ENGRAMDB_SC4_ROWS", "16"))
    _STATE["store"] = os.environ.get("ENGRAMDB_SC4_STORE")
    _STATE["counters"] = os.environ.get("ENGRAMDB_SC4_COUNTERS")
    if arm in ("read", "read_static"):
        if not _STATE["store"]:
            raise RuntimeError(f"arm={arm} needs ENGRAMDB_SC4_STORE")
        _open_store(_STATE["store"])

    _HOSTBUF = np.zeros((4096, 4096), dtype=np.uint8)  # reused, host-only

    if arm == "read_static":
        # Fixed token window, rowids resolved once.  Nothing in the break then
        # touches the device except the H2D of the delta, so any remaining cost
        # is the disk read plus the add.
        _STATE["static_toks"] = [1000 + (7 * i) % 200000 for i in range(4096)]
        _STATE["static_rowids"] = _rowids(np.asarray(_STATE["static_toks"],
                                                     dtype=np.int64))

    # --- proof layer 1: did the graph replay at all? -----------------------
    _orig_backend_replay = BreakableCudaGraphBackend.replay

    def _backend_replay(self, *a, **kw):
        with _lock:
            _C["backend_replay_calls"] += 1
        return _orig_backend_replay(self, *a, **kw)

    BreakableCudaGraphBackend.replay = _backend_replay

    # --- proof layer 2: was our body inside that replay loop? --------------
    _orig_graph_replay = BreakableCUDAGraph.replay

    def _graph_replay(self, *a, **kw):
        global _IN_REPLAY
        with _lock:
            _C["bcg_replay_calls"] += 1
        prev, _IN_REPLAY = _IN_REPLAY, True
        try:
            return _orig_graph_replay(self, *a, **kw)
        finally:
            _IN_REPLAY = prev

    BreakableCUDAGraph.replay = _graph_replay

    # --- the break body ---------------------------------------------------
    @eager_on_graph(True)
    def _ple_break(hidden_states, input_ids):
        """Runs inline outside capture, and as host Python between segments at
        replay.  Mutates ``hidden_states`` in place and returns None.

        Returning None is the documented contract: ``_copy_output`` is
        per-tensor, not per-tuple, so a break that must write back has to
        mutate a buffer it was handed (see ``models/inkling.py``, which does
        exactly this and says so).
        """
        global _IN_REPLAY
        t0 = time.perf_counter()
        try:
            n = hidden_states.shape[0]
            if _STATE["arm"] == "read_static":
                # No D2H: rowids were resolved at install time.
                toks = _STATE["static_toks"][:n]
                _delta_from_disk(toks, _STATE["static_rowids"][:n],
                                 _HOSTBUF[:n, :hidden_states.shape[-1]])
            elif _STATE["arm"] == "read":
                toks = input_ids[:n].tolist()      # D2H: serial, and declared --
                rowids = _rowids(toks)             # stage 1 is deliberately the
                _delta_from_disk(toks, rowids,     # non-overlapped variant.
                                 _HOSTBUF[:n, :hidden_states.shape[-1]])
            else:
                toks = input_ids[:n].tolist()
                _delta_from_tokens(toks, _STATE["rows"],
                                   _HOSTBUF[:n, :hidden_states.shape[-1]])
            delta = torch.from_numpy(
                _HOSTBUF[:n, :hidden_states.shape[-1]].copy()
            ).to(device=hidden_states.device, dtype=hidden_states.dtype)
            hidden_states.add_(delta)
            with _lock:
                _C["break_calls"] += 1
                if _IN_REPLAY:
                    _C["break_calls_in_replay"] += 1
                else:
                    _C["break_calls_in_capture"] += 1
                if len(_C["break_us"]) < _KEEP:
                    _C["break_us"].append((time.perf_counter() - t0) * 1e6)
                if len(_C["tok_samples"]) < 6:
                    _C["tok_samples"].append(toks[:8])
        except Exception as exc:                    # never kill the graph silently
            with _lock:
                _C["errors"].append(f"{type(exc).__name__}: {exc}")
            raise

    # --- splice it after the chosen layer's forward ------------------------
    target = _STATE["layer"]

    def _wrap(cls):
        orig = cls.forward

        def forward(self, *args, **kwargs):
            out = orig(self, *args, **kwargs)
            # Attribute check first: the wrapper is on every decoder layer, and
            # only one of them is ours.  Counting before the check would put a
            # lock acquisition on 24 extra calls per decode step.
            if getattr(self, "layer_id", None) != target:
                return out
            with _lock:
                _C["forward_calls"] += 1
                nm = type(self).__name__
                _C["forward_by_class"][nm] = _C["forward_by_class"].get(nm, 0) + 1
            hs = out[0] if isinstance(out, tuple) else out
            fb = kwargs.get("forward_batch")
            if fb is None:
                for a in args:
                    if type(a).__name__ == "ForwardBatch":
                        fb = a
                        break
            ids = getattr(fb, "input_ids", None)
            if torch.is_tensor(hs) and torch.is_tensor(ids):
                _ple_break(hs, ids)
            else:
                with _lock:
                    if len(_C["skipped"]) < 5:
                        _C["skipped"].append(
                            f"layer_id={getattr(self,'layer_id',None)} "
                            f"hs={type(hs).__name__} ids={type(ids).__name__} "
                            f"fb={type(fb).__name__}"
                        )
            return out

        cls.forward = forward
        with _lock:
            _C["wrapped_classes"].append(cls.__name__)

    _wrap(qwen3_5.Qwen3_5LinearDecoderLayer)
    _wrap(qwen3_5.Qwen3_5AttentionDecoderLayer)

    _install_dump()
    _INSTALLED = True
    print(
        f"[engramdb-sc4] arm={arm} layer={target} rows={_STATE['rows']} "
        f"store={_STATE['store']}"
    )
    return True


def _rowids(toks):
    import numpy as np
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from ref_ple_hash import head_vocab_sizes, ple_rowids, spec_from_weights

    if not hasattr(_rowids, "_mult"):
        _rowids._mult = spec_from_weights(
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "docs")
        )
        # Cache the prime table too: head_vocab_sizes() trial-divides 16 numbers
        # near 2e7 on *every* call, which is milliseconds of host time per decode
        # step and would be charged to the break.
        _rowids._sizes = head_vocab_sizes()
    return ple_rowids(np.asarray(toks, dtype=np.int64), _rowids._mult,
                      _rowids._sizes)


# ---------------------------------------------------------------------------
# counters out
# ---------------------------------------------------------------------------

def _dump(*_a) -> None:
    """Write this process's counters to ``<base>.<pid>.json``.

    Per-pid on purpose.  SGLang imports the model module in *several* processes
    -- the tokenizer manager builds the config, the scheduler runs the model --
    so they all install this hook and all reach the dump.  A shared path means
    the last writer wins, and the tokenizer's all-zero counters silently
    overwrite the scheduler's real ones.  That is not hypothetical: it is what
    the first smoke run of this probe reported.
    """
    path = _STATE.get("counters")
    if not path:
        return
    path = f"{path}.{os.getpid()}.json"
    with _lock:
        snap = dict(_C)
    for k in ("read_us", "break_us"):
        v = snap.get(k) or []
        snap[k + "_n"] = len(v)
        snap[k + "_median"] = sorted(v)[len(v) // 2] if v else None
        snap[k + "_mean"] = (sum(v) / len(v)) if v else None
        snap[k] = v[:40]
    snap["arm"] = _STATE["arm"]
    snap["layer"] = _STATE["layer"]
    snap["rows"] = _STATE["rows"]
    snap["pid"] = os.getpid()
    snap["process"] = _proc_name()
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(snap, f, indent=2)
    os.replace(tmp, path)


def _proc_name() -> str:
    try:
        with open("/proc/self/cmdline", "rb") as f:
            return " ".join(f.read().decode(errors="replace").split("\0"))[:400]
    except Exception:
        return "?"


def _install_dump() -> None:
    """Counters must survive a SIGKILL.

    ``engine.shutdown()`` tears the scheduler down with signals that cannot be
    caught, so ``atexit`` and signal handlers both lose the race and the child
    writes nothing -- the first two smoke runs of this probe reported all-zero
    counters for exactly that reason.  A background thread that snapshots the
    in-memory dicts every 500 ms is unaffected: the break path only ever touches
    memory, and the I/O happens off the critical path, so the timing this probe
    exists to measure is not perturbed by the act of recording it.
    """
    atexit.register(_dump)
    for sig in (signal.SIGTERM, signal.SIGQUIT, signal.SIGUSR1, signal.SIGINT):
        try:
            signal.signal(sig, lambda s, f: (_dump(), os._exit(0)))
        except Exception:
            pass
    if _STATE.get("counters"):
        t = threading.Thread(target=_flush_loop, daemon=True, name="sc4-flush")
        t.start()


def _flush_loop() -> None:
    while True:
        time.sleep(0.5)
        try:
            _dump()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(0 if install() else 0)
