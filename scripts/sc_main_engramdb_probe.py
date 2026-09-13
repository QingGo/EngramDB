#!/usr/bin/env python3
"""SGLang `main` + the EngramDB store: real PLE geometry, our disk store, in a graph.

Session 45 proved the *seam* (SGLang main's own `Qwen4ExpPLELayer` → `gather` →
`BreakableCUDAGraph`) with upstream's sparse-file mapping as the row source.
This probe keeps the seam and replaces the row source with `engramdb.Store`,
using the **real** Qwen3.8-Flash-Next PLE geometry and the **real** 51.2 GB
table (128 shards x 2,500,012 rows x 160 B) rather than a scaled-down stand-in.

    PYTHONPATH=<sglang-main>/python python sc_main_engramdb_probe.py \
        --sections geometry,ids,store,graph

Sections
  geometry  build the real `Qwen4ExpTextConfig` from the checkpoint config, on the
            meta device so a 51.2 GB vocab-parallel weight is never materialised;
            assert the engine's padded vocab == the store's row count, and that the
            engine's per-layer multipliers == the constants our rowid path uses.
  ids       the engine's own `_hash_contexts` vs `engramdb.rowids_for_seq` on the
            same tokens. If these disagree, nothing downstream can be right.
  store     `Store.fetch` vs an independent `pread` of the same raw shard files.
  graph     the real breakable graph, rows served by the store, replay freshness.

Deliberately *not* stand-ins: the PLE config, the table, the row source, the seam.
Still stand-ins: the model body (never instantiated) and the "preceding decoder
layer" (one matmul). τ(1) is Session 44's.
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

DEFAULT_CFG = (
    "/root/autodl-tmp/qwen35-ple/models/Qwen3.8-Flash-Next-FP8-tokenizer/config.json"
)
DEFAULT_STORE = "/root/autodl-tmp/qwen35-ple/qwen38-rows"


# --------------------------------------------------------------------------
# setup
# --------------------------------------------------------------------------
def _init_runtime(args):
    """Publish main's runtime context and stand up one rank."""
    from sglang.srt.runtime_context import get_parallel, publish
    from sglang.srt.server_args import ServerArgs

    publish(ServerArgs(model_path=args.model_path, tp_size=1), role="test")
    if get_parallel().tp_size != 1:
        raise RuntimeError("expected tp_size 1")

    import sglang.srt.distributed as dist

    dist.init_distributed_environment(
        world_size=1,
        rank=0,
        local_rank=0,
        distributed_init_method="tcp://127.0.0.1:29531",
        backend="nccl",
    )
    dist.initialize_model_parallel(tensor_model_parallel_size=1)


def _stub_pool(q):
    """`_hash_contexts` reaches for the global req-to-token pool only to cache
    its shift tables, and that pool needs a live attention backend."""

    class _Pool:
        ple_window_cache = None

    q.get_req_to_token_pool = lambda: _Pool()


def build_real_config(q, args):
    """The checkpoint's own text_config, so every PLE constant is the real one.

    One deliberate deviation: the checkpoint reaches fp8 through its
    `quantization_config` (`_ple_table_is_fp8` accepts either that or a declared
    `ple_embedding_dtype`), and building the real quant config pulls in the
    model-loading stack for no benefit here. Declaring the dtype picks the same
    `params_dtype`, and the row width is asserted against the store below.
    """
    raw = json.load(open(args.config))["text_config"]
    cfg = q.Qwen4ExpTextConfig(
        **raw,
        ple_embedding_dtype="float8_e4m3fn",
        ple_offload_embedding=True,
        ple_offload_backend="engramdb",
        ple_offload_dir=args.store,
    )
    return cfg


class RawShardOracle:
    """Rows straight out of `shard_NNN.bin` with `pread`. No engramdb code."""

    def __init__(self, root: str, shards: int, rows_per_shard: int, width: int):
        self.fds = [
            os.open(os.path.join(root, f"shard_{i:03d}.bin"), os.O_RDONLY)
            for i in range(shards)
        ]
        self.rows_per_shard = int(rows_per_shard)
        self.width = int(width)

    def rows(self, ids) -> torch.Tensor:
        out = torch.empty((len(ids), self.width), dtype=torch.uint8)
        for k, r in enumerate(ids):
            shard, off = divmod(int(r), self.rows_per_shard)
            buf = os.pread(self.fds[shard], self.width, off * self.width)
            if len(buf) != self.width:
                raise RuntimeError(f"short pread for row {r}: {len(buf)}")
            out[k] = torch.frombuffer(bytearray(buf), dtype=torch.uint8)
        return out

    def close(self):
        for fd in self.fds:
            os.close(fd)


def open_store(args):
    import engramdb

    manifest = json.load(open(os.path.join(args.store, "manifest.json")))
    shards = int(manifest["num_shards"])
    shard_bytes = int(manifest["expected_shard_bytes"])
    # width comes from the checkpoint geometry, not from the manifest; row count
    # follows from it and is checked against the engine in the geometry section.
    width = args.head_dim
    rows_per_shard = shard_bytes // width
    store = engramdb.Store(
        args.store,
        shards=shards,
        rows_per_shard=rows_per_shard,
        width=width,
        threads=args.threads,
    )
    return store, shards, rows_per_shard, width


class StoreEmbedding:
    """Factory for the probe's embedding class (built once `q` is imported)."""

    @staticmethod
    def make(q, table_dtype, staging_dtype=None, device_cast=False):
        import torch.nn as nn

        from sglang.srt.layers.quantization.unquant import UnquantizedEmbeddingMethod

        class _StoreEmbedding(q.Qwen4ExpStagedFileEmbedding):
            """`file-staged` with the row source swapped for an EngramDB store.

            Inherits the eager `gather` (and therefore the graph break) from the
            patch; only `__init__` (no host table to allocate) and `_read_rows`
            (read from the store instead of a mapping) differ. Patch 0002 will
            move this into the tree; here it lives in the probe so the design can
            still move.
            """

            def __init__(
                self, embedding, store, store_rows: int, staging_dtype=None,
                device_cast=False,
            ):
                nn.Module.__init__(self)
                if not isinstance(embedding.quant_method, UnquantizedEmbeddingMethod):
                    raise NotImplementedError(
                        "PLE embedding offload requires an unquantized embedding table"
                    )
                if embedding.weight.dtype not in (
                    torch.bfloat16,
                    torch.float8_e4m3fn,
                ):
                    raise TypeError(
                        f"unsupported PLE table dtype {embedding.weight.dtype}"
                    )
                if embedding.num_added_embeddings:
                    raise NotImplementedError(
                        "PLE embedding offload does not support added vocabulary rows"
                    )
                for name in q.Qwen4ExpPinnedHostEmbedding._COPIED_ATTRIBUTES:
                    setattr(self, name, getattr(embedding, name))
                self.quant_method = None
                self._table_dtype = embedding.weight.dtype
                # Staging in the table's own dtype lets one pinned->device copy do
                # the fp8->bf16 cast on the GPU. The CPU cast is 13-15x slower and
                # scales with rows; the GPU's is flat.
                self._staging_dtype = staging_dtype or self._table_dtype
                self._device_cast = device_cast
                self._dev_stage = None
                self._store = store
                self._store_rows = int(store_rows)
                self._file_prefetcher = None
                self._staging = None
                self._staging_rows = 0
                self.register_buffer(
                    "weight_scale",
                    torch.ones(1, dtype=torch.bfloat16),
                    persistent=True,
                )
                del embedding.weight

            def _staging_slab(self, rows: int) -> torch.Tensor:
                if self._staging_rows < rows:
                    self._staging_rows = max(rows, self._STAGING_ROWS_FLOOR)
                    self._staging = torch.empty(
                        (self._staging_rows, self.embedding_dim),
                        dtype=self._staging_dtype,
                        pin_memory=True,
                    )
                return self._staging[:rows]

            def _rows_from_bytes(self, raw: bytes, n: int) -> torch.Tensor:
                row_bytes = self.embedding_dim * torch.empty(
                    0, dtype=self._table_dtype
                ).element_size()
                if len(raw) != n * row_bytes:
                    raise RuntimeError(
                        f"store returned {len(raw)} B for {n} rows of {row_bytes} B"
                    )
                return (
                    torch.frombuffer(bytearray(raw), dtype=torch.uint8)
                    .view(self._table_dtype)
                    .view(-1, self.embedding_dim)
                )

            def _publish(self, staged: torch.Tensor, output: torch.Tensor) -> None:
                """fp8 staging -> device, cast on the device.

                One cross-device copy that also casts costs 409 us at 2048 rows;
                two same-dtype copies with the cast on the device cost ~60. The
                cast is 18 us on the GPU either way, so paying it inside `copy_`
                is what is expensive, not the cast.
                """
                n = staged.shape[0]
                dev = self._dev_stage
                if dev is None or dev.shape[0] < n:
                    dev = torch.empty(
                        (max(n, self._STAGING_ROWS_FLOOR), self.embedding_dim),
                        dtype=self._table_dtype,
                        device=output.device,
                    )
                    self._dev_stage = dev
                dev[:n].copy_(staged, non_blocking=False)
                output.view(-1, self.embedding_dim).copy_(dev[:n])

            @q.eager_on_graph(True, capture_stub=q._ple_staged_capture_stub)
            def gather(
                self, input_ids: torch.Tensor, out: Optional[torch.Tensor] = None
            ) -> torch.Tensor:
                output = self._resolve_gather_output(input_ids, out)
                flat_ids = input_ids.reshape(-1).long()
                if flat_ids.numel():
                    staged = self._read_rows(flat_ids.detach().cpu())
                    if self._device_cast:
                        self._publish(staged, output)
                    else:
                        output.view(-1, self.embedding_dim).copy_(staged)
                return output

            def _read_rows(self, host_ids: torch.Tensor) -> torch.Tensor:
                """Rows from the store, as a pinned slab of the table's dtype.

                Same contract as the mapping version -- row-major, shard masking
                applied, staging reused -- but the bytes come from `Store.fetch`
                (one native batched gather) instead of `index_select` over an
                mmap, and the slab stays fp8 so the single pinned->device copy in
                `gather` performs the cast on the GPU.
                """
                start = int(self.shard_indices.org_vocab_start_index)
                end = int(self.shard_indices.org_vocab_end_index)
                n = host_ids.numel()
                staged = self._staging_slab(n)
                in_shard = (host_ids >= start) & (host_ids < end)
                if bool(in_shard.all()):
                    # The common single-store case: no masking, no zeroing.
                    staged.copy_(
                        self._rows_from_bytes(
                            self._store.fetch((host_ids - start).tolist()), n
                        )
                    )
                else:
                    staged.zero_()
                    if bool(in_shard.any()):
                        keep = in_shard.nonzero(as_tuple=True)[0]
                        local = (host_ids[keep] - start).tolist()
                        staged[keep] = self._rows_from_bytes(
                            self._store.fetch(local), len(local)
                        )
                return staged

        def build(embedding, store, store_rows: int):
            return _StoreEmbedding(
                embedding,
                store,
                store_rows,
                staging_dtype=staging_dtype,
                device_cast=device_cast,
            )

        return build


# --------------------------------------------------------------------------
# sections
# --------------------------------------------------------------------------
def section_geometry(q, args, store, shards, rows_per_shard, width):
    cfg = build_real_config(q, args)
    report = {
        "ple_embed_dim": cfg.ple_embed_dim,
        "ngram_size": cfg.ngram_size,
        "heads_per_ngram": cfg.heads_per_ngram,
        "ngram_vocab_size_base": cfg.ngram_vocab_size_base,
        "vocab_size": cfg.vocab_size,
        "eos_token_id": cfg.eos_token_id,
        "ple_layer_ids": cfg.ple_layer_ids,
        "shards": shards,
        "rows_per_shard": rows_per_shard,
        "width": width,
        "store_rows": int(store.total_rows),
        "store_width": int(store.width),
    }
    # Build on meta: with the real base the vocab-parallel weight is 51.2 GB and
    # we only want its shape and the hash constants.
    with torch.device("meta"):
        emb = q.Qwen4ExpNGramEmbedding(cfg, cfg.ple_embed_dim, ple_layer_index=0)
    vp = emb.ngram_embedding
    # Meta built the hash buffers as meta tensors too; resynthesise them with the
    # module's own builders so `_hash_contexts` can actually run.
    emb.layer_multipliers = emb._build_layer_multipliers(emb.ngram_size).cuda()
    _sizes, _offsets, _total = emb._build_head_vocab_and_offsets()
    emb.ngram_heads_vocab_sizes = torch.tensor(
        _sizes, dtype=torch.long, device="cuda"
    )
    emb.ngram_heads_offsets = torch.tensor(_offsets, dtype=torch.long, device="cuda")
    report["padded_vocab_size"] = int(vp.num_embeddings)
    report["num_embeddings_per_partition"] = int(vp.num_embeddings_per_partition)
    report["head_dim_per_ngram"] = int(emb.head_dim_per_ngram)
    report["ngram_heads"] = int(emb.ngram_heads)
    report["padded_vocab_equals_store_rows"] = (
        int(vp.num_embeddings) == int(store.total_rows)
    )
    report["width_equals_head_dim"] = int(vp.embedding_dim) == int(store.width)
    engine_row_bytes = int(vp.embedding_dim) * torch.empty(
        0, dtype=vp.weight.dtype
    ).element_size()
    report["table_dtype"] = str(vp.weight.dtype)
    report["engine_row_bytes"] = engine_row_bytes
    report["row_bytes_equals_store_width"] = engine_row_bytes == int(store.width)

    # The engine's constants vs the ones our native rowid path hardcodes.
    import engramdb

    from engramdb.ple_adapter import ple_rowids  # noqa: F401  (reference path)

    multipliers = [int(x) for x in emb._build_layer_multipliers(emb.ngram_size)]
    report["engine_layer_multipliers"] = multipliers
    # rowids_for_seq with no explicit multipliers must equal the engine's.
    proto = list(range(1000, 1000 + emb.ngram_size))
    via_native = engramdb.rowids_for_seq(proto + list(range(2000, 2016)))
    via_engine_mult = engramdb.rowids_for_seq(
        proto + list(range(2000, 2016)), multipliers=multipliers
    )
    report["native_rowids_match_engine_multipliers"] = via_native == via_engine_mult

    sizes, offsets, total = emb._build_head_vocab_and_offsets()
    report["head_vocab_total"] = int(total)
    report["ngram_heads_vocab_sizes_head"] = [int(s) for s in sizes[:4]]
    report["ngram_heads_offsets_head"] = [int(o) for o in offsets[:4]]
    return emb, cfg, report


def section_ids(q, emb, cfg, args):
    """Engine hash vs our rowid function, on the same tokens."""
    emb.enable_ple_fusion = False  # portable path on both devices
    g = torch.Generator().manual_seed(args.seed)
    seq = torch.randint(0, cfg.vocab_size, (args.seq_len,), generator=g).tolist()
    pad = [cfg.eos_token_id] * (emb.ngram_size - 1)
    windows = [pad + seq[: i + 1] for i in range(len(seq))]
    contexts = torch.tensor([w[-emb.ngram_size :] for w in windows], dtype=torch.long)

    engine_ids = emb._hash_contexts(contexts.to("cuda"), decode_sized=True).cpu()
    our_ids = torch.tensor(
        __import__("engramdb").rowids_for_seq(seq), dtype=torch.long
    )
    equal = bool(torch.equal(engine_ids, our_ids))
    report = {
        "seq_len": len(seq),
        "equal": equal,
        "engine_ids_row0": [int(x) for x in engine_ids[0].tolist()],
        "our_ids_row0": [int(x) for x in our_ids[0].tolist()],
        "mismatch_rows": int((engine_ids != our_ids).any(dim=1).sum()),
        "max_abs_diff": int((engine_ids - our_ids).abs().max()),
    }
    if not equal:
        bad = torch.nonzero((engine_ids != our_ids).any(dim=1)).flatten()[:5]
        report["first_mismatch"] = [
            {
                "i": int(i),
                "engine": [int(x) for x in engine_ids[i].tolist()],
                "ours": [int(x) for x in our_ids[i].tolist()],
            }
            for i in bad
        ]
    return report


def section_store(store, args, shards, rows_per_shard, width):
    """Store.fetch vs an independent pread of the same rows."""
    g = torch.Generator().manual_seed(args.seed)
    ids = torch.randint(0, int(store.total_rows), (args.rows,), generator=g).tolist()
    oracle = RawShardOracle(args.store, shards, rows_per_shard, width)
    want = oracle.rows(ids)
    got = torch.frombuffer(bytearray(store.fetch(ids)), dtype=torch.uint8).view(
        args.rows, width
    )
    eq = bool(torch.equal(got, want))
    report = {
        "rows": args.rows,
        "equal_to_pread": eq,
        "distinct_rows": len({bytes(r) for r in got.numpy()}),
        "zero_frac": float((got == 0).float().mean()),
    }
    if not eq:
        diff = (got != want).any(dim=1)
        report["mismatch_rows"] = int(diff.sum())
        report["first_mismatch_rowid"] = int(ids[int(torch.nonzero(diff)[0])])
    # cold cost: drop both the page cache and any mapping we hold
    os.sync()

    def timed(fn, n):
        ts = []
        for _ in range(n):
            t0 = time.perf_counter()
            fn()
            ts.append((time.perf_counter() - t0) * 1e6)
        return {"median": statistics.median(ts), "min": min(ts), "max": max(ts)}

    fresh = [
        torch.randint(0, int(store.total_rows), (args.rows,), generator=g).tolist()
        for _ in range(args.iters)
    ]

    def cold(fn):
        ts = []
        for i in range(args.iters):
            fds = oracle.fds
            for fd in fds:
                try:
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                except OSError:
                    pass
            t0 = time.perf_counter()
            fn(fresh[i])
            ts.append((time.perf_counter() - t0) * 1e6)
        return {"median": statistics.median(ts), "min": min(ts), "max": max(ts)}

    report["fetch_warm_us"] = timed(lambda: store.fetch(ids), args.iters)
    report["fetch_cold_us"] = cold(store.fetch)
    report["pread_cold_us"] = cold(oracle.rows)
    oracle.close()
    return report


def section_graph(q, args, cfg, store, store_rows, EmbeddingCls):
    """The real breakable graph, rows from the store."""
    from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.breakable_cuda_graph import (
        BreakableCUDAGraph,
        BreakableCUDAGraphCapture,
    )

    if args.real_layer:
        # The tree's own selection path: Qwen4ExpPLELayer picks the backend from
        # the config, so this is the integration rather than a probe stand-in.
        with torch.device("meta"):
            layer = q.Qwen4ExpPLELayer(
                cfg,
                quant_config=None,
                prefix="model.layers.1",
                layer_id=1,
                ple_layer_index=0,
            )
        inner = layer.ple_embedding
        assert isinstance(
            inner.ngram_embedding, q.Qwen4ExpEngramDbEmbedding
        ), f"backend not selected: {type(inner.ngram_embedding).__name__}"
    else:
        with torch.device("meta"):
            inner = q.Qwen4ExpNGramEmbedding(cfg, cfg.ple_embed_dim, ple_layer_index=0)
    # Meta built only for the shapes; give it real hash buffers so the engine's
    # own `_hash_contexts` runs, then swap the embedding for the store-backed one.
    inner.layer_multipliers = inner._build_layer_multipliers(inner.ngram_size)
    sizes, offsets, _ = inner._build_head_vocab_and_offsets()
    inner.ngram_heads_vocab_sizes = torch.tensor(sizes, dtype=torch.long)
    inner.ngram_heads_offsets = torch.tensor(offsets, dtype=torch.long)
    inner.enable_ple_fusion = False
    if not args.real_layer:
        inner.ngram_embedding = EmbeddingCls(inner.ngram_embedding, store, store_rows)
    emb = inner.ngram_embedding

    heads = inner.ngram_heads  # owned by the outer module, not the offloaded one
    ids_buf = torch.zeros((args.tokens, heads), dtype=torch.long, device="cuda")
    out_buf = emb.allocate_output(
        (args.tokens, heads, emb.embedding_dim), torch.device("cuda")
    )
    consume_buf = torch.zeros_like(out_buf)
    w = torch.randn(512, 512, device="cuda") / 32
    x = torch.randn(args.tokens, 512, device="cuda")
    work_buf = torch.zeros(args.tokens, 512, device="cuda")
    torch.mm(x, w)
    torch.cuda.synchronize()

    graph = BreakableCUDAGraph()
    capture_stream = torch.cuda.Stream()
    with BreakableCUDAGraphCapture(graph, pool=(0, 0), stream=capture_stream):
        work_buf.copy_(torch.mm(x, w))
        gathered = emb.gather(ids_buf, out=out_buf)
        consume_buf.copy_(gathered)

    # Replay correctness against the independent pread oracle. Kept in its own
    # loop: the oracle reads the table too, so interleaving it with timing would
    # churn the page cache the timing is trying to measure.
    manifest = json.load(open(os.path.join(args.store, "manifest.json")))
    grad = args.head_dim
    rows_per_shard = int(manifest["expected_shard_bytes"]) // grad
    oracle = RawShardOracle(
        args.store, int(manifest["num_shards"]), rows_per_shard, grad
    )
    fresh = [
        torch.randint(0, store_rows, (args.tokens, heads), dtype=torch.long)
        for _ in range(max(args.iters, args.correct_iters))
    ]
    os.sync()
    ok = 0
    mismatches = []
    for f in fresh[: args.correct_iters]:
        ids_buf.copy_(f)
        graph.replay()
        torch.cuda.synchronize()
        want = oracle.rows(f.reshape(-1).tolist()).view(torch.float8_e4m3fn)
        got = consume_buf.view(-1, args.head_dim).cpu()
        if torch.equal(got, want.to(torch.bfloat16)):
            ok += 1
        else:
            mismatches.append(int((got != want.to(torch.bfloat16)).sum()))

    fixed_early = fresh[0]

    def drop_cache(ids):
        """Evict only the pages this step will read.

        Dropping the whole 51.2 GB table first would leave the kernel still
        walking it while the read starts, and that contention -- not the storage
        -- dominates the number. A row is 160 B inside one 4 KiB page, so the
        surgical drop is exact and does not race the read.
        """
        per_shard: dict[int, list[int]] = {}
        for r in ids:
            shard, off = divmod(int(r), rows_per_shard)
            per_shard.setdefault(shard, []).append(off * args.head_dim)
        for shard, offs in per_shard.items():
            fd = oracle.fds[shard]
            for off in offs:
                try:
                    os.posix_fadvise(
                        fd, (off // 4096) * 4096, 4096, os.POSIX_FADV_DONTNEED
                    )
                except OSError:
                    pass

    def timed(ids):
        ts = []
        for one in ids:
            ids_buf.copy_(one)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            graph.replay()
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1e6)
        return ts

    # Eager `gather` on the same buffers, so the graph machinery's own share is
    # the difference rather than an assumption.
    eager = []
    for _ in range(args.iters):
        ids_buf.copy_(fixed_early)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        emb.gather(ids_buf, out=out_buf)
        torch.cuda.synchronize()
        eager.append((time.perf_counter() - t0) * 1e6)

    # Same process, same buffers: split the eager gather into its parts. cProfile
    # was useless here -- its numbers contradicted the wall-clock ones by 7x on
    # the C calls, because the store's worker threads and the tracer fight over
    # the GIL.
    host_ids = fixed_early.reshape(-1).long().cpu()
    slab = emb._read_rows(host_ids)
    parts = {
        "d2h_ids": timed([fixed_early.reshape(-1).long().cpu()] * args.iters)[0]
        if False
        else None,
    }
    def one(fn):
        ts = []
        for _ in range(args.iters):
            t0 = time.perf_counter()
            fn()
            ts.append((time.perf_counter() - t0) * 1e6)
        return statistics.median(ts)

    parts = {
        "d2h_ids": one(lambda: fixed_early.reshape(-1).long().cpu()),
        "read_rows": one(lambda: emb._read_rows(host_ids)),
        "publish_only": one(
            lambda: (
                emb._publish(slab, out_buf)
                if getattr(emb, "_device_cast", True)
                else out_buf.view(-1, emb.embedding_dim).copy_(slab)
            )
        ),
        "publish_legacy": one(
            lambda: out_buf.view(-1, emb.embedding_dim).copy_(slab)
        ),
        "gather_all": one(lambda: emb.gather(ids_buf, out=out_buf)),
    }
    print(
        f"PARTS cls={type(emb).__name__} "
        f"device_cast={getattr(emb, '_device_cast', 'native')} " + json.dumps(parts),
        flush=True,
    )

    if args.profile:
        import cProfile
        import io
        import pstats

        pr = cProfile.Profile()
        pr.enable()
        for _ in range(args.iters):
            emb.gather(ids_buf, out=out_buf)
        torch.cuda.synchronize()
        pr.disable()
        buf = io.StringIO()
        pstats.Stats(pr, stream=buf).sort_stats("tottime").print_stats(20)
        print(buf.getvalue(), flush=True)

    # Warm: one fixed id set, replayed until the rows are resident.
    fixed = fresh[0]
    for _ in range(3):
        ids_buf.copy_(fixed)
        graph.replay()
    torch.cuda.synchronize()
    warm = timed([fixed] * args.iters)

    # Cold: new random rows each replay, page cache dropped first.
    cold = []
    for one in fresh:
        drop_cache(one.reshape(-1).tolist())
        ids_buf.copy_(one)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        graph.replay()
        torch.cuda.synchronize()
        cold.append((time.perf_counter() - t0) * 1e6)
    oracle.close()

    return {
        "segments": len(graph._segments),
        "break_fns": len(graph._break_fns),
        "tokens": args.tokens,
        "rows_per_step": args.tokens * heads,
        "replay_reads_fresh_rows": ok,
        "replays_checked": args.correct_iters,
        "mismatch_elems": mismatches[:5],
        "parts_us": parts,
        "eager_us_median": statistics.median(eager),
        "eager_us_min": min(eager),
        "warm_us_median": statistics.median(warm),
        "warm_us_min": min(warm),
        "warm_us_max": max(warm),
        "cold_us_median": statistics.median(cold),
        "cold_us_min": min(cold),
        "cold_us_max": max(cold),
    }


def section_phases(q, args, cfg, store, store_rows):
    """Where the staged step's time actually goes, and what each fix would buy.

    Session 44's lesson: attribute before optimising. The step is
    d2h(ids) + store read + bytes->tensor + publish(h2d). The candidate fixes are
    (a) a different thread count, (b) skipping the `bytearray` copy that
    `torch.frombuffer` forces on read-only `bytes`, and (c) converting fp8->bf16
    on the device instead of the host -- 160 B/row over PCIe instead of 320, and
    a vectorised convert. Each is timed alone.
    """
    import numpy as np
    import engramdb

    g = torch.Generator().manual_seed(args.seed)
    out = {"rows": {}, "threads": {}}

    def timed(fn, n):
        ts = []
        for _ in range(n):
            t0 = time.perf_counter()
            fn()
            ts.append((time.perf_counter() - t0) * 1e6)
        return {"median": statistics.median(ts), "min": min(ts)}

    for tokens in args.phase_tokens:
        rows = tokens * 16
        ids = torch.randint(0, store_rows, (tokens, 16), generator=g)
        flat = ids.reshape(-1).tolist()
        ids_dev = ids.reshape(-1).long().cuda()
        raw = store.fetch(flat)
        row = {}
        row["d2h_ids"] = timed(lambda: ids_dev.cpu(), args.iters)
        row["fetch"] = timed(lambda: store.fetch(flat), args.iters)
        row["bytes_to_fp8"] = timed(
            lambda: torch.frombuffer(bytearray(raw), dtype=torch.uint8)
            .view(torch.float8_e4m3fn)
            .view(-1, args.head_dim),
            args.iters,
        )
        row["numpy_view_fp8"] = timed(
            lambda: torch.from_numpy(np.frombuffer(raw, dtype=np.uint8)).view(
                torch.float8_e4m3fn
            ).view(-1, args.head_dim),
            args.iters,
        )
        src = (
            torch.frombuffer(bytearray(raw), dtype=torch.uint8)
            .view(torch.float8_e4m3fn)
            .view(-1, args.head_dim)
        )
        row["cpu_fp8_to_bf16"] = timed(lambda: src.to(torch.bfloat16), args.iters)
        pinned_bf16 = src.to(torch.bfloat16).pin_memory()
        dev_fp8 = torch.empty((rows, args.head_dim), dtype=torch.float8_e4m3fn, device="cuda")
        dev_bf16 = torch.empty((rows, args.head_dim), dtype=torch.bfloat16, device="cuda")
        row["h2d_fp8"] = timed(lambda: dev_fp8.copy_(src), args.iters)
        row["h2d_bf16_pinned"] = timed(lambda: dev_bf16.copy_(pinned_bf16), args.iters)
        row["gpu_fp8_to_bf16"] = timed(
            lambda: dev_bf16.copy_(dev_fp8.to(torch.bfloat16)), args.iters
        )
        row["publish_h2d"] = timed(lambda: dev_bf16.copy_(pinned_bf16), args.iters)
        out["rows"][f"tokens={tokens}"] = {k: v["median"] for k, v in row.items()}
        print(
            f"    tokens={tokens:>4} ({rows:>5} rows) "
            + "  ".join(f"{k}={v['median']:.1f}" for k, v in row.items()),
            flush=True,
        )
        del ids_dev, dev_fp8, dev_bf16, src, pinned_bf16

    for threads in args.threads_sweep:
        st = engramdb.Store(
            args.store,
            shards=int(store.total_rows) // 2500012,
            rows_per_shard=2500012,
            width=args.head_dim,
            threads=threads,
        )
        for tokens in (1, 8, 128):
            flat = torch.randint(0, store_rows, (tokens * 16,), generator=g).tolist()
            st.fetch(flat)
            out["threads"][f"t={threads},tok={tokens}"] = timed(
                lambda: st.fetch(flat), args.iters
            )["median"]
        st.close()
    return out


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sections", default="geometry,ids,store,graph")
    ap.add_argument("--config", default=DEFAULT_CFG)
    ap.add_argument("--store", default=DEFAULT_STORE)
    ap.add_argument("--model-path", default="/root/autodl-tmp/qwen35-ple/models/Qwen3.5-0.8B")
    ap.add_argument("--head-dim", type=int, default=160)
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--rows", type=int, default=2048)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--correct-iters", type=int, default=5)
    ap.add_argument("--real-layer", action="store_true",
                    help="build Qwen4ExpPLELayer and let the tree select the backend")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--staging", choices=["fp8", "bf16", "devfp8"], default="devfp8",
                    help="staging slab dtype; fp8 moves the cast to the device")
    ap.add_argument("--phase-tokens", default="1,8,32,128")
    ap.add_argument("--threads-sweep", default="1,2,4,8,16,32")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    args.sections = [s for s in args.sections.split(",") if s]
    args.phase_tokens = [int(x) for x in args.phase_tokens.split(",") if x]
    args.threads_sweep = [int(x) for x in args.threads_sweep.split(",") if x]

    import sglang

    print(f"sglang from {os.path.dirname(sglang.__file__)}", flush=True)
    from sglang.srt.models import qwen4_exp as q

    _init_runtime(args)
    _stub_pool(q)

    report = {"argv": sys.argv[1:], "gpu": torch.cuda.get_device_name(0)}
    store, shards, rows_per_shard, width = open_store(args)
    print(
        f"store: {shards} shards x {rows_per_shard} rows x {width} B "
        f"= {store.total_rows} rows",
        flush=True,
    )
    state = {}
    for name in args.sections:
        try:
            if name == "geometry":
                emb, cfg, r = section_geometry(
                    q, args, store, shards, rows_per_shard, width
                )
                state["emb"], state["cfg"] = emb, cfg
                report["geometry"] = r
            elif name == "ids":
                report["ids"] = section_ids(q, state["emb"], state["cfg"], args)
            elif name == "store":
                report["store"] = section_store(
                    store, args, shards, rows_per_shard, width
                )
            elif name == "phases":
                report["phases"] = section_phases(
                    q, args, state["cfg"], store, int(store.total_rows)
                )
            elif name == "graph":
                cls = StoreEmbedding.make(
                    q,
                    torch.float8_e4m3fn,
                    staging_dtype=(
                        torch.float8_e4m3fn
                        if args.staging in ("fp8", "devfp8")
                        else torch.bfloat16
                    ),
                    device_cast=args.staging == "devfp8",
                )
                report["graph"] = section_graph(
                    q, args, state["cfg"], store, int(store.total_rows), cls
                )
        except Exception:
            report[name] = {"error": traceback.format_exc()}
            print(f"!!! section {name} failed:\n{traceback.format_exc()}", flush=True)
        if name in report:
            print(f"SUMMARY {name}: {json.dumps(report[name])[:1500]}", flush=True)
    if args.out:
        json.dump(report, open(args.out, "w"), indent=2)
        print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
