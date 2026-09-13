#!/usr/bin/env python3
"""SGLang-main PLE offload probe: file-backed table access on a non-HMM GPU.

Runs against a *source checkout* of sgl-project/sglang (main):

    PYTHONPATH=<checkout>/python python sc_main_staged_probe.py --sections setup,eager,graph

so the seam under test is main's own ``Qwen4ExpNGramEmbedding``,
``Qwen4ExpPinnedHostEmbedding`` / ``Qwen4ExpStagedFileEmbedding`` and main's own
``BreakableCUDAGraph`` -- not a re-implementation of them.

Arms
  pinned   upstream device-side gather out of pinned host RAM. Reference for
           correctness, and the timing floor: no host round trip.
  direct   upstream ``file`` backend: the Triton kernel dereferences a pageable
           mapping through the device. On a non-HMM part that is exactly what
           ``check_file_backend_supported`` refuses to load. Run this arm in its
           own process; a segfault or a CUDA error is a result, not a bug.
  staged   the patch: the host reads the mapping into pinned staging and one
           pinned->device copy publishes it; ``gather`` is an eager break.

The measurement we care about is whether a *cold* read can be issued early
enough to stay off the decode critical path. ``start_prefetch`` for the PLE
layer at model layer i+1 is called before layer i runs, so the lead time is
exactly one decoder layer, and the budget is tau(1). What this probe reports is
the *exposure*: how much of the gather lands inside the replayed step.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import json
import os
import statistics
import sys
import time
import traceback
from typing import Optional

import torch


# --------------------------------------------------------------------------
# environment
# --------------------------------------------------------------------------
def _probe_import():
    import sglang

    root = os.path.dirname(os.path.dirname(os.path.abspath(sglang.__file__)))
    print(f"sglang imported from {root}", flush=True)
    from sglang.srt.models import qwen4_exp as q

    have = [n for n in dir(q) if "Staged" in n]
    print(f"qwen4_exp from {q.__file__}", flush=True)
    print(f"staged classes present: {have or 'NONE (patch not applied)'}", flush=True)
    return q


def _free_port() -> int:
    import socket

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _init_parallel(args):
    """One rank, one GPU, and a published runtime context.

    main's ``VocabParallelEmbedding`` reads ``tp_size`` from the process-wide
    runtime context rather than from a module global, so the config has to be
    published even though no engine is started; ``role="test"`` is the
    enumerated role for exactly that. The model path only has to be a readable
    checkpoint -- nothing is loaded from it.
    """
    from sglang.srt.runtime_context import get_parallel, publish
    from sglang.srt.server_args import ServerArgs

    publish(ServerArgs(model_path=args.model_path, tp_size=1), role="test")
    if get_parallel().tp_size != 1:
        raise RuntimeError(f"expected tp_size 1, got {get_parallel().tp_size}")

    import sglang.srt.distributed as dist

    if hasattr(dist, "init_distributed_environment"):
        dist.init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{_free_port()}",
            backend="nccl",
        )
    if hasattr(dist, "initialize_model_parallel"):
        dist.initialize_model_parallel(tensor_model_parallel_size=1)


# --------------------------------------------------------------------------
# the table
# --------------------------------------------------------------------------
def _row_bytes(dim: int, dtype: torch.dtype) -> int:
    return dim * torch.empty(0, dtype=dtype).element_size()


def fill_table(table: torch.Tensor, seed: int = 0) -> dict:
    """Deterministically fill an fp8/bf16 host table and describe it.

    fp8 E4M3 byte 0x7F/0xFF are NaN; keeping the top nibble below 7 avoids them
    so byte-equality comparisons in the checks are meaningful.
    """
    rows, dim = table.shape
    g = torch.Generator().manual_seed(seed)
    chunk = 1 << 20
    if table.dtype == torch.float8_e4m3fn:
        for i in range(0, rows, chunk):
            n = min(chunk, rows - i)
            block = torch.randint(0, 0x70, (n, dim), generator=g, dtype=torch.uint8)
            table[i : i + n].copy_(block.view(torch.float8_e4m3fn))
    else:
        for i in range(0, rows, chunk):
            n = min(chunk, rows - i)
            table[i : i + n].copy_(
                torch.randn((n, dim), generator=g, dtype=torch.float32).to(table.dtype)
            )
    # The rows above are dirty in the mapping; a later fadvise(DONTNEED) cannot
    # evict a dirty page, so push them out before anything is timed.
    os.sync()
    sample = table[: min(rows, 4096)].clone()
    raw = sample.view(torch.uint8)
    distinct = len({r.numpy().tobytes() for r in raw})
    try:
        zero_frac = float((sample.to(torch.float32) == 0).float().mean())
    except Exception:
        zero_frac = None
    return {
        "rows": int(rows),
        "dim": int(dim),
        "dtype": str(table.dtype),
        "row_bytes": _row_bytes(dim, table.dtype),
        "bytes": int(rows * _row_bytes(dim, table.dtype)),
        "distinct_rows_in_first_4096": distinct,
        "zero_frac": zero_frac,
    }


class HostReference:
    """Independent oracle: a second, private mapping of the same file."""

    def __init__(self, path: str, shape, dtype: torch.dtype, start: int, end: int):
        nbytes = shape[0] * shape[1] * torch.empty(0, dtype=dtype).element_size()
        raw = torch.from_file(path, shared=False, size=nbytes, dtype=torch.uint8)
        self.table = raw.view(dtype).view(*shape)
        self.start, self.end = int(start), int(end)
        self.dim = int(shape[1])

    def rows(self, host_ids: torch.Tensor) -> torch.Tensor:
        """Row-major over any id shape, matching the gather's flat layout."""
        flat = host_ids.reshape(-1)
        out = torch.zeros((flat.numel(), self.dim), dtype=torch.bfloat16)
        m = (flat >= self.start) & (flat < self.end)
        if bool(m.any()):
            local = (flat[m] - self.start).to(torch.long)
            out[m] = self.table.index_select(0, local).to(torch.bfloat16)
        return out


def drop_caches(path: str, addr: int, nbytes: int) -> None:
    """Make the next read cold: page cache and the mapping's resident pages."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)
    libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True)
    libc.madvise(ctypes.c_void_p(addr), ctypes.c_size_t(nbytes), ctypes.c_int(4))


# --------------------------------------------------------------------------
# module construction
# --------------------------------------------------------------------------
def build_config(q, args):
    """A tiny Qwen4Exp text config that keeps the real PLE geometry.

    head dim 160 and 16 heads per token are the shipped model's numbers, so a
    row is the same 160 B and a gather the same 16 rows/token. Only the table's
    row count is scaled down; the model body is never instantiated.
    """
    ple_embed_dim = args.head_dim * args.heads_per_ngram * (args.ngram_size - 1)
    heads = args.heads_per_ngram * (args.ngram_size - 1)
    elem = 1 if args.dtype == "fp8" else 2
    # table bytes = 16 rows/token * base rows/head * (ple_embed_dim/16) B
    base = max(int(args.table_mb * 2**20 / (ple_embed_dim * elem)), 1000)
    return q.Qwen4ExpTextConfig(
        vocab_size=4096,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        intermediate_size=512,
        hc_count=4,
        ple_layer_ids=[2],
        ple_embed_dim=ple_embed_dim,
        ngram_size=args.ngram_size,
        heads_per_ngram=args.heads_per_ngram,
        ngram_vocab_size_base=base,
        make_ngram_vocab_size_divisible_by=128,
        eos_token_id=2,
        ple_offload_embedding=True,
        ple_offload_backend="pinned",
        ple_offload_dir=args.table_dir,
        ple_embedding_dtype="float8_e4m3fn" if args.dtype == "fp8" else None,
    ), base


def build_offloaded(q, config, arm: str, args):
    """The real module chain, with the real swap the PLE layer does."""
    inner = q.Qwen4ExpNGramEmbedding(config, config.ple_embed_dim, ple_layer_index=0)
    backend = {
        "pinned": "pinned",
        "direct": "file",
        "staged": "file-staged",
        "staged-nobreak": "file-staged",
    }[arm]
    cls = (
        q.Qwen4ExpStagedFileEmbedding
        if arm.startswith("staged")
        else q.Qwen4ExpPinnedHostEmbedding
    )
    offloaded = cls(inner.ngram_embedding, backend=backend, table_dir=args.table_dir)
    inner.ngram_embedding = offloaded
    return inner, offloaded


def table_path_of(offloaded, table_dir: str, arm: str) -> Optional[str]:
    """Rebuild the table's path, or None when the table is not in a file.

    The ``arm`` guard matters: ``ple_table_file_name`` is deterministic from
    shape and tag, so a *pinned* arm would otherwise resolve a file left behind
    by an earlier file-backed run of the same shape and "cold" it -- dropping
    cache against an unrelated file, and calling madvise on pinned RAM.

    ``allocate_ple_host_table`` tags the tensor it returns, but the embedding
    wraps that tensor in an ``nn.Parameter`` and only re-copies attributes off
    the *original* weight, so the tag is gone by the time the module is built.
    Recomputing is safe precisely because the name is deterministic.
    """
    if arm == "pinned":
        return None
    from sglang.srt.models.qwen4_exp_ple_table import ple_table_file_name

    si = offloaded.shard_indices
    name = ple_table_file_name(
        tuple(offloaded.weight.shape),
        offloaded.weight.dtype,
        f"rows{si.org_vocab_start_index}-{si.org_vocab_end_index}",
    )
    path = os.path.join(os.path.expanduser(table_dir), name)
    return path if os.path.exists(path) else None


# --------------------------------------------------------------------------
# sections
# --------------------------------------------------------------------------
def section_setup(q, args):
    config, base = build_config(q, args)
    print(
        f"config: ple_embed_dim={config.ple_embed_dim} heads="
        f"{(config.ngram_size - 1) * config.heads_per_ngram} "
        f"vocab_base={base} dtype={args.dtype}",
        flush=True,
    )
    inner, offloaded = build_offloaded(q, config, "staged", args)
    path = table_path_of(offloaded, args.table_dir, "staged")
    print(f"staged table path: {path}", flush=True)
    stats = fill_table(offloaded.weight.data, seed=args.seed)
    print("table: " + json.dumps(stats), flush=True)
    ids = torch.randint(
        0, int(offloaded.weight.shape[0]), (args.tokens, 16), dtype=torch.long
    )
    got = offloaded.gather(ids.to("cuda")).cpu()
    ref = HostReference(
        path,
        tuple(offloaded.weight.shape),
        offloaded.weight.dtype,
        offloaded.shard_indices.org_vocab_start_index,
        offloaded.shard_indices.org_vocab_end_index,
    ).rows(ids)
    same = bool(torch.equal(got.view(-1, offloaded.embedding_dim), ref))
    print(f"eager staged gather == host reference: {same}", flush=True)
    return {"eager_equal": same, "table": stats, "path": path, "vocab_base": base}


def section_eager(q, args):
    """Per-phase cost of one staged gather, warm and cold."""
    config, _ = build_config(q, args)
    out = {}
    for arm in args.arms:
        if arm == "direct":
            continue
        inner, offloaded = build_offloaded(q, config, arm, args)
        path = table_path_of(offloaded, args.table_dir, arm)
        if path:
            fill_table(offloaded.weight.data, seed=args.seed)
        ids = torch.randint(
            0, int(offloaded.weight.shape[0]), (args.tokens, 16), dtype=torch.long
        )
        ids_dev = ids.to("cuda")
        out_dev = offloaded.allocate_output(
            (*ids.shape, offloaded.embedding_dim), torch.device("cuda")
        )
        for _ in range(args.warmup):
            offloaded.gather(ids_dev, out=out_dev)
        torch.cuda.synchronize()
        rows = {}

        def timed(fn, n):
            ts = []
            for _ in range(n):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                fn()
                torch.cuda.synchronize()
                ts.append((time.perf_counter() - t0) * 1e6)
            return {"median": statistics.median(ts), "min": min(ts), "max": max(ts)}

        rows["step_warm"] = timed(lambda: offloaded.gather(ids_dev, out=out_dev), args.iters)
        if path:
            nbytes = offloaded.weight.numel() * offloaded.weight.element_size()
            addr = offloaded.weight.data_ptr()

            def cold_step():
                drop_caches(path, addr, nbytes)
                offloaded.gather(ids_dev, out=out_dev)

            rows["step_cold"] = timed(cold_step, args.iters)
        if arm == "staged":
            host_ids = ids.reshape(-1).long()
            rows["d2h_ids"] = timed(lambda: ids_dev.reshape(-1).long().cpu(), args.iters)
            rows["read_rows_warm"] = timed(lambda: offloaded._read_rows(host_ids.clone()), args.iters)

            def cold_read():
                drop_caches(path, addr, nbytes)
                offloaded._read_rows(host_ids.clone())

            rows["read_rows_cold"] = timed(cold_read, args.iters)
            staged = offloaded._read_rows(host_ids.clone())
            dst = out_dev.view(-1, offloaded.embedding_dim)
            rows["h2d_publish"] = timed(lambda: dst.copy_(staged), args.iters)
        out[arm] = rows
        print(f"--- eager {arm} ({args.tokens} tok x 16 rows = {args.tokens * 16} rows)")
        for k, v in rows.items():
            print(
                f"    {k:16s} median {v['median']:8.1f} us   min {v['min']:8.1f}   max {v['max']:8.1f}"
            )
        sys.stdout.flush()
    return out


def section_graph(q, args):
    """Replay the real breakable graph and check what replay actually reads."""
    from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
        breakable_cuda_graph as bcg,
    )
    from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.breakable_cuda_graph import (
        BreakableCUDAGraph,
        BreakableCUDAGraphCapture,
    )

    config, _ = build_config(q, args)
    heads = (config.ngram_size - 1) * config.heads_per_ngram
    results = {}
    for arm in args.arms:
        inner, offloaded = build_offloaded(q, config, arm, args)
        # "staged-nobreak" is the same class with the decorator's capture hook
        # disarmed, i.e. what happens if the graph backend is not breakable.
        saved_var = None
        if arm == "staged-nobreak":
            saved_var = _disarm_eager_on_graph(bcg)
        try:
            results[arm] = _graph_arm(
                q, args, offloaded, arm, heads, BreakableCUDAGraph,
                BreakableCUDAGraphCapture,
            )
        except Exception:
            results[arm] = {"error": traceback.format_exc()}
            print(f"--- graph {arm} FAILED:\n{traceback.format_exc()}", flush=True)
        finally:
            if saved_var is not None:
                bcg._current_capture_var = saved_var
        torch.cuda.empty_cache()
    return results


def _disarm_eager_on_graph(bcg):
    """Make ``eager_on_graph`` inert, the way a non-breakable backend does.

    Replacing ``_current_capture_var`` with a fresh ContextVar does *not* work:
    ``BreakableCUDAGraphCapture.__enter__`` sets that same global, so reader and
    writer move together. The var has to keep reporting "not capturing" while
    the capture still sets it, which is what this does.
    """
    saved = bcg._current_capture_var
    bcg._current_capture_var = _NeverCapturing()
    return saved


class _NeverCapturing:
    """Duck-typed stand-in for the capture ContextVar, always unset.

    Not a ``ContextVar`` subclass -- the C type is not an acceptable base -- and
    only the three methods the module calls are needed.
    """

    def get(self, default=None):
        return None

    def set(self, value):
        return None

    def reset(self, token):
        return None


def _graph_arm(q, args, offloaded, arm, heads, BreakableCUDAGraph, Capture):
    path = table_path_of(offloaded, args.table_dir, arm)
    if path:
        fill_table(offloaded.weight.data, seed=args.seed)
    ref = (
        HostReference(
            path,
            tuple(offloaded.weight.shape),
            offloaded.weight.dtype,
            offloaded.shard_indices.org_vocab_start_index,
            offloaded.shard_indices.org_vocab_end_index,
        )
        if path
        else None
    )

    ids_buf = torch.zeros((args.tokens, heads), dtype=torch.long, device="cuda")
    out_buf = offloaded.allocate_output(
        (args.tokens, heads, offloaded.embedding_dim), torch.device("cuda")
    )
    consume_buf = torch.zeros_like(out_buf)
    # A stand-in for the decoder layer whose work the read should overlap.
    w = torch.randn(512, 512, device="cuda") / 32
    x = torch.randn(args.tokens, 512, device="cuda")
    work_buf = torch.zeros(args.tokens, 512, device="cuda")
    # cuBLAS creates its handle on first use, which capture forbids; the engine
    # pays that in its own warmup runs before capturing, so pay it here too.
    torch.mm(x, w)
    torch.cuda.synchronize()

    graph = BreakableCUDAGraph()
    # The engine captures on a dedicated stream; CUDA refuses the default one.
    # Replay is legal on any stream, so the timed loop stays on the default.
    capture_stream = torch.cuda.Stream()
    with Capture(graph, pool=(0, 0), stream=capture_stream):
        work_buf.copy_(torch.mm(x, w))
        gathered = offloaded.gather(ids_buf, out=out_buf)
        consume_buf.copy_(gathered)
    n_breaks = len(graph._break_fns)
    n_segments = len(graph._segments)

    fresh = [
        torch.randint(
            0, int(offloaded.weight.shape[0]), (args.tokens, heads), dtype=torch.long
        )
        for _ in range(args.iters)
    ]
    nbytes = offloaded.weight.numel() * offloaded.weight.element_size()
    addr = offloaded.weight.data_ptr()

    times, fresh_ok = [], []
    ids_buf.copy_(fresh[0])
    graph.replay()  # warm: first replay pays lazy graph instantiation
    torch.cuda.synchronize()
    for f in fresh:
        ids_buf.copy_(f)
        if args.cold and path:
            drop_caches(path, addr, nbytes)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        graph.replay()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e6)
        if ref is not None:
            want = ref.rows(f)
            got = consume_buf.view(-1, offloaded.embedding_dim).cpu()
            fresh_ok.append(bool(torch.equal(got, want)))

    r = {
        "break_fns": n_breaks,
        "segments": n_segments,
        "cold": bool(args.cold and path),
        "step_us_median": statistics.median(times),
        "step_us_min": min(times),
        "step_us_max": max(times),
        "replay_reads_fresh_rows": sum(1 for v in fresh_ok if v) if ref is not None else None,
        "replays_checked": len(fresh_ok) if ref is not None else 0,
    }
    print(
        f"--- graph {arm}: segments={n_segments} breaks={n_breaks} "
        f"cold={r['cold']} step={r['step_us_median']:.1f} us "
        f"(min {r['step_us_min']:.1f} max {r['step_us_max']:.1f})",
        flush=True,
    )
    if ref is not None:
        print(
            f"    replay read fresh rows: {r['replay_reads_fresh_rows']}/{r['replays_checked']}",
            flush=True,
        )
    del graph
    return r


def section_direct(q, args):
    """The upstream `file` backend on a device that cannot address it.

    Deliberately unguarded: this is what `--ple-offload-backend file` would do
    with SGLANG_QWEN4_PLE_FILE_SKIP_DEVICE_CHECK=1, i.e. what the load-time gate
    exists to prevent.
    """
    config, _ = build_config(q, args)
    inner, offloaded = build_offloaded(q, config, "direct", args)
    path = table_path_of(offloaded, args.table_dir, "direct")
    stats = fill_table(offloaded.weight.data, seed=args.seed)
    ids = torch.randint(
        0, int(offloaded.weight.shape[0]), (args.tokens, 16), dtype=torch.long
    )
    ref = HostReference(
        path,
        tuple(offloaded.weight.shape),
        offloaded.weight.dtype,
        offloaded.shard_indices.org_vocab_start_index,
        offloaded.shard_indices.org_vocab_end_index,
    ).rows(ids)
    got = offloaded.gather(ids.to("cuda")).cpu()
    torch.cuda.synchronize()
    got = got.view(-1, offloaded.embedding_dim)
    return {
        "device_attr_host_page_tables": _host_page_tables_attr(),
        "equal_to_reference": bool(torch.equal(got, ref)),
        "max_abs_diff": float((got.float() - ref.float()).abs().max()),
        "nonzero_where_ref_zero": int(((ref == 0) & (got != 0)).sum()),
        "reference_nonzero": int((ref != 0).sum()),
        "table": stats,
    }


def _host_page_tables_attr():
    try:
        from sglang.srt.models.qwen4_exp_ple_table import (
            device_uses_host_page_tables,
            ple_file_access_mode,
        )

        return {
            "uses_host_page_tables": device_uses_host_page_tables(0),
            "access_mode": ple_file_access_mode(0),
        }
    except Exception as exc:  # pragma: no cover - diagnostic only
        return {"error": repr(exc)}


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sections", default="setup,eager,graph")
    ap.add_argument("--arms", default="pinned,staged")
    ap.add_argument(
        "--model-path",
        default="/root/autodl-tmp/qwen35-ple/models/Qwen3.5-0.8B",
        help="readable checkpoint dir; only used to build ServerArgs for publish()",
    )
    ap.add_argument("--table-dir", default="/root/engramdb_probe/tables")
    ap.add_argument("--table-mb", type=float, default=256.0)
    ap.add_argument("--dtype", choices=["fp8", "bf16"], default="fp8")
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--ngram-size", type=int, default=3)
    ap.add_argument("--heads-per-ngram", type=int, default=8)
    ap.add_argument("--head-dim", type=int, default=160)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cold", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    args.arms = [a for a in args.arms.split(",") if a]
    args.sections = [s for s in args.sections.split(",") if s]
    os.makedirs(args.table_dir, exist_ok=True)

    report = {"argv": sys.argv[1:], "gpu": torch.cuda.get_device_name(0)}
    q = _probe_import()
    _init_parallel(args)
    if "direct" in args.arms:
        report["direct"] = section_direct(q, args)  # may kill the process
        args.arms = [a for a in args.arms if a != "direct"]
    for name in args.sections:
        fn = {"setup": section_setup, "eager": section_eager, "graph": section_graph}[name]
        try:
            report[name] = fn(q, args)
        except Exception:
            report[name] = {"error": traceback.format_exc()}
            print(f"!!! section {name} failed:\n{traceback.format_exc()}", flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"wrote {args.out}", flush=True)
    for name in ("setup", "eager", "graph", "direct"):
        if name in report:
            print(f"SUMMARY {name}: {json.dumps(report[name])}", flush=True)


if __name__ == "__main__":
    main()
