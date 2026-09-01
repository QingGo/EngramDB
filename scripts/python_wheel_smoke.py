#!/usr/bin/env python3
"""Smoke-test an installed engramdb-python wheel.

This intentionally avoids PyTorch/engram-peft so it can run in a plain Python
environment on every wheel platform. It exercises the native extension plus the
SGLang/vLLM-facing helpers, the multi-table Database, Arrow helpers, and the
minimal TCP service added in 0.2.5.
"""

from __future__ import annotations

import json
import os
import socket
import tempfile
import threading
from pathlib import Path

import engramdb


def _make_table(root: Path, name: str, rows: int = 4, width: int = 8) -> None:
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "shard_000.bin", "wb") as f:
        for i in range(rows):
            f.write(bytes([i % 256]) * width)


def test_page_reader() -> None:
    readers = [
        name
        for name in ("PageReader", "IoUringPageReader")
        if getattr(engramdb, name, None) is not None
    ]
    if getattr(engramdb, "PageReader", None) is not None:
        readers.append("SGLangPageReader")
    if not readers:
        print("No page reader available on this platform; skipping")
        return

    page_size = 16
    payload = bytes(range(page_size))
    tmp = tempfile.NamedTemporaryFile(prefix="engramdb-page-", delete=False)
    tmp.write(payload)
    tmp.close()

    try:
        fd = os.open(tmp.name, os.O_RDONLY)
        try:
            for name in readers:
                from engramdb.sglang import SGLangPageReader
                if name == "SGLangPageReader":
                    reader = SGLangPageReader(page_size=page_size)
                else:
                    reader = getattr(engramdb, name)(page_size=page_size)
                pages = reader.read_pages([fd], [0])
                assert len(pages) == 1
                assert pages[0] == payload
                print(f"{name} OK")
        finally:
            os.close(fd)
    finally:
        os.unlink(tmp.name)


def test_slot_index() -> None:
    if engramdb.SlotIndex is None:
        print("SlotIndex skipped: numpy not available")
        return

    import numpy as np

    rowids = np.arange(32, dtype=np.int64).reshape(2, 16)
    index = engramdb.SlotIndex.from_rowids(rowids)
    assert index.lookup(tuple(range(16, 32))) == 1
    with tempfile.TemporaryDirectory(prefix="engramdb-slot-") as td:
        path = Path(td) / "index.npz"
        index.save(path)
        restored = engramdb.SlotIndex.load(path)
        assert restored.lookup(tuple(range(16))) == 0
    print("SlotIndex OK")


def test_disk_slot_index() -> None:
    if engramdb.DiskSlotIndex is None:
        print("DiskSlotIndex skipped: numpy not available")
        return

    import numpy as np

    rowids = np.arange(64, dtype=np.int64).reshape(4, 16)
    with tempfile.TemporaryDirectory(prefix="engramdb-disk-slot-") as td:
        idx = engramdb.DiskSlotIndex.build(rowids, td, num_buckets=8)
        try:
            assert len(idx) == 4
            assert idx.lookup(tuple(range(16))) == 0
            assert idx.lookup(tuple(range(16, 32))) == 1
            slots = idx.to_slots(rowids)
            np.testing.assert_array_equal(slots, np.arange(4, dtype=np.int64))
        finally:
            idx.close()
    with tempfile.TemporaryDirectory(prefix="engramdb-disk-slot-v2-") as td:
        idx = engramdb.DiskSlotIndex.build(
            rowids, td, num_buckets=8, hash_name="fnv1a-64"
        )
        try:
            assert idx.hash_name == "fnv1a-64"
            assert idx.lookup(tuple(range(16))) == 0
            assert idx.lookup(tuple(range(32, 48))) == 2
        finally:
            idx.close()
    with tempfile.TemporaryDirectory(prefix="engramdb-disk-slot-v3-") as td:
        idx = engramdb.DiskSlotIndex.build(
            rowids, td, num_buckets=8, hash_name="fnv1a-64", single_file=True
        )
        try:
            assert idx.format == "engramdb-disk-slot-index-v3"
            assert idx.lookup(tuple(range(16))) == 0
            assert idx.lookup(tuple(range(16, 32))) == 1
            slots = idx.to_slots(rowids)
            np.testing.assert_array_equal(slots, np.arange(4, dtype=np.int64))
        finally:
            idx.close()
    print("DiskSlotIndex OK")


def test_store_and_vllm_gather() -> None:
    row_width = 8
    rows = [bytes([i] * row_width) for i in range(4)]
    with tempfile.TemporaryDirectory(prefix="engramdb-store-") as directory:
        with open(os.path.join(directory, "shard_000.bin"), "wb") as f:
            for row in rows:
                f.write(row)

        store = engramdb.Store(directory, 1, len(rows), row_width)
        try:
            assert store.width == row_width
            raw = store.fetch([2, 0, 2, 1])
            assert raw == rows[2] + rows[0] + rows[2] + rows[1]

            from engramdb.vllm import PleDiskGather

            gather = PleDiskGather(store, row_bytes=row_width)
            expanded = gather.fetch([2, 0, 2, 1])
            assert expanded == rows[2] + rows[0] + rows[2] + rows[1]
            unique = gather.fetch_unique([2, 0, 2, 1])
            assert unique == rows[2] + rows[0] + rows[1]
            print("Store + PleDiskGather OK")
        finally:
            store.close()



def test_store_concurrent_fetch() -> None:
    from concurrent.futures import ThreadPoolExecutor

    row_width = 8
    rows = [bytes([i] * row_width) for i in range(64)]
    with tempfile.TemporaryDirectory(prefix="engramdb-concurrent-") as directory:
        with open(os.path.join(directory, "shard_000.bin"), "wb") as f:
            for row in rows:
                f.write(row)
        store = engramdb.Store(directory, 1, len(rows), row_width)
        try:
            def fetch_all(_: int) -> int:
                return len(store.fetch(list(range(64))))

            with ThreadPoolExecutor(max_workers=4) as ex:
                results = list(ex.map(fetch_all, range(8)))
            assert results == [64 * row_width] * 8
            print("Store concurrent fetch OK")
        finally:
            store.close()


def test_database_arrow_server() -> None:
    from engramdb import Database
    from engramdb.server import EngramDBServer

    with tempfile.TemporaryDirectory(prefix="engramdb-smoke-") as td:
        root = Path(td)
        _make_table(root, "alpha")
        _make_table(root, "beta")

        db = Database(root)
        assert db.list_tables() == ["alpha", "beta"], db.list_tables()
        raw = db.fetch("alpha", [1, 3], shards=1, rows_per_shard=4, width=8)
        expected = bytes([1] * 8) + bytes([3] * 8)
        assert raw == expected, (raw, expected)
        print("Database OK:", db.list_tables())

        # Optional Arrow helpers.
        try:
            from engramdb.arrow_utils import store_fetch_arrow, table_to_ipc_bytes

            store = db.open_store("alpha", 1, 4, 8)
            try:
                table = store_fetch_arrow(store, [0, 2])
                ipc = table_to_ipc_bytes(table)
                assert table.num_rows == 2
                assert table.column_names == ["rowid", "row"]
                assert len(ipc) > 0
                print("Arrow OK:", table.num_rows, table.column_names, "ipc_bytes", len(ipc))
            finally:
                store.close()
        except ImportError:
            print("Arrow skipped: pyarrow not available")

        # Minimal TCP/JSON service.
        server = EngramDBServer(db, host="127.0.0.1", port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        host, port = server.server_address

        def call(req: dict) -> dict:
            with socket.create_connection((host, port), timeout=5) as sock:
                sock.sendall((json.dumps(req) + "\n").encode("utf-8"))
                return json.loads(sock.makefile("rb").readline())

        ping = call({"cmd": "ping"})
        assert ping["ok"] and ping["pong"]
        tables = call({"cmd": "list_tables"})
        assert tables["tables"] == ["alpha", "beta"]
        fet = call({
            "cmd": "fetch",
            "table": "alpha",
            "rowids": [0, 2],
            "shards": 1,
            "rows_per_shard": 4,
            "width": 8,
        })
        assert fet["ok"]
        import base64
        assert base64.b64decode(fet["raw_base64"]) == db.fetch(
            "alpha", [0, 2], shards=1, rows_per_shard=4, width=8
        )
        print("Server OK: ping/list_tables/fetch")

        server.shutdown()
        server.server_close()
        db.close()


def test_disk_ple_lru() -> None:
    try:
        import torch
    except Exception:
        print("DiskPleEmbedding LRU skipped: torch not available")
        return

    from engramdb.vllm_plugin import DiskPleEmbedding

    # Each float32 PLE row is 4 logical floats = 16 raw bytes on disk.
    raw_width = 16
    logical_dim = 4
    rows = [bytes([i] * raw_width) for i in range(8)]
    with tempfile.TemporaryDirectory(prefix="engramdb-lru-") as directory:
        with open(os.path.join(directory, "shard_000.bin"), "wb") as f:
            for row in rows:
                f.write(row)
        store = engramdb.Store(directory, 1, len(rows), raw_width)
        try:
            emb = DiskPleEmbedding(
                store,
                num_embeddings=len(rows),
                embedding_dim=logical_dim,
                dtype=torch.float32,
                cache_size=4,
            )
            # A single 3-token lookup exercises miss + LRU fill + cache hit.
            indices = torch.tensor([2, 0, 2, 1, 2])
            out = emb(indices)
            assert tuple(out.shape) == (5, logical_dim)
            assert out.dtype == torch.float32
            assert len(emb._cache) <= emb.cache_size
            # The cache should contain a subset of the accessed rows.
            assert set(emb._cache.keys()).issubset({0, 1, 2})
            print("DiskPleEmbedding LRU OK:", tuple(out.shape), "cache", len(emb._cache))
        finally:
            store.close()


def test_prefetch_lru() -> None:
    try:
        import torch
    except Exception:
        print("DiskPleEmbedding prefetch skipped: torch not available")
        return

    from engramdb.vllm_plugin import DiskPleEmbedding

    raw_width = 4
    rows = [bytes([i] * raw_width) for i in range(8)]
    with tempfile.TemporaryDirectory(prefix="engramdb-prefetch-") as directory:
        with open(os.path.join(directory, "shard_000.bin"), "wb") as f:
            for row in rows:
                f.write(row)
        store = engramdb.Store(directory, 1, len(rows), raw_width)
        try:
            emb = DiskPleEmbedding(
                store,
                num_embeddings=len(rows),
                embedding_dim=1,
                dtype=torch.float32,
                cache_size=4,
            )
            fut = emb.prefetch([2, 0, 1])
            assert fut is not None
            out = emb(torch.tensor([2, 0, 2, 1]))
            assert tuple(out.shape) == (4, 1)
            stats = emb.get_stats()
            assert stats["prefetch_issued"] == 3.0
            assert stats["misses"] == 0.0
            assert set(emb._cache.keys()).issubset({0, 1, 2})
            emb.close()
            assert emb._closed
            print("DiskPleEmbedding prefetch OK")
        finally:
            store.close()


def test_fetch_e_t_tensor() -> None:
    try:
        import torch
    except Exception:
        print("fetch_e_t_tensor skipped: torch not available")
        return

    import struct

    from engramdb.vllm import PleDiskGather, fetch_e_t_tensor

    # 4 rows, each one float32 (a single head dimension of width 1).
    rows = [struct.pack("<f", float(i)) for i in range(4)]
    with tempfile.TemporaryDirectory(prefix="engramdb-tensor-") as directory:
        with open(os.path.join(directory, "shard_000.bin"), "wb") as f:
            for row in rows:
                f.write(row)
        store = engramdb.Store(directory, 1, len(rows), 4)
        try:
            # Flat head rows: token0 = [row2, row0], token1 = [row2, row1].
            flat = [2, 0, 2, 1]
            out = fetch_e_t_tensor(
                store,
                flat,
                scale=2.0,
                num_heads=2,
                head_dim=1,
                dtype=torch.float32,
                out_dtype=torch.float32,
            )
            assert tuple(out.shape) == (2, 2, 1)
            assert out[0, 0, 0].item() == 4.0
            assert out[0, 1, 0].item() == 0.0
            assert out[1, 1, 0].item() == 2.0

            # PleDiskGather.fetch must return exact original-order bytes and
            # should also support the direct tensor path.
            raw = PleDiskGather(store, 4).fetch(flat)
            assert raw == rows[2] + rows[0] + rows[2] + rows[1]
            out2 = PleDiskGather(store, 4).fetch_tensor(
                flat,
                scale=2.0,
                num_heads=2,
                head_dim=1,
                dtype=torch.float32,
                out_dtype=torch.float32,
            )
            assert torch.equal(out, out2)
            print("fetch_e_t_tensor OK")
        finally:
            store.close()


def test_disk_ple_nocache_fastpath() -> None:
    try:
        import torch
    except Exception:
        print("DiskPle no-cache fast path skipped: torch not available")
        return

    import struct

    from engramdb.vllm_plugin import DiskPleEmbedding

    rows = [struct.pack("<f", float(i)) for i in range(4)]
    with tempfile.TemporaryDirectory(prefix="engramdb-nocache-") as directory:
        with open(os.path.join(directory, "shard_000.bin"), "wb") as f:
            for row in rows:
                f.write(row)
        store = engramdb.Store(directory, 1, len(rows), 4)
        try:
            emb = DiskPleEmbedding(
                store,
                num_embeddings=4,
                embedding_dim=1,
                dtype=torch.float32,
                cache_size=0,
            )
            out = emb(torch.tensor([2, 0, 2, 1]))
            assert tuple(out.shape) == (4, 1)
            assert out[0, 0].item() == 2.0
            assert out[1, 0].item() == 0.0
            stats = emb.get_stats()
            assert stats["misses"] == 4.0
            assert stats["fetch_s"] >= 0.0
            emb.close()
            assert emb._closed
            print("DiskPle no-cache fast path OK")
        finally:
            store.close()



def test_safetensors_i64_reader() -> None:
    import json
    import struct

    from engramdb.ple_discovery import read_safetensors_i64

    multipliers = [23_703_573_157_769, 20_109_073_645_365, 8_052_911_324_071]
    payload = struct.pack("<3q", *multipliers)
    header = {
        "__metadata__": {},
        "layer_multipliers": {
            "dtype": "I64",
            "shape": [3],
            "data_offsets": [0, len(payload)],
        },
    }
    header_bytes = json.dumps(header).encode("utf-8")
    with tempfile.TemporaryDirectory(prefix="engramdb-i64-") as td:
        path = Path(td) / "tensor.safetensors"
        with open(path, "wb") as f:
            f.write(struct.pack("<Q", len(header_bytes)))
            f.write(header_bytes)
            f.write(payload)
        got = read_safetensors_i64(path, "layer_multipliers")
        assert got == multipliers, (got, multipliers)
    print("read_safetensors_i64 OK")


def test_discover_ple_metadata() -> None:
    import json
    import struct

    from engramdb import discover_ple, load_ple_multipliers, load_ple_weight_scale

    multipliers = [23_703_573_157_769, 20_109_073_645_365, 8_052_911_324_071]
    scale = 0.00019931793212890625
    scale_bytes = struct.pack("<f", scale)
    mult_bytes = struct.pack("<3q", *multipliers)
    shard_bytes = bytes(160)
    payload = scale_bytes + mult_bytes + shard_bytes
    header = {
        "__metadata__": {},
        "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.weight_scale": {
            "dtype": "F32",
            "shape": [1],
            "data_offsets": [0, len(scale_bytes)],
        },
        "model.language_model.layers.1.ple.ple_embedding.layer_multipliers": {
            "dtype": "I64",
            "shape": [3],
            "data_offsets": [len(scale_bytes), len(scale_bytes) + len(mult_bytes)],
        },
        "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight": {
            "dtype": "F8_E4M3",
            "shape": [1, 160],
            "data_offsets": [
                len(scale_bytes) + len(mult_bytes),
                len(payload),
            ],
        },
    }
    header_bytes = json.dumps(header).encode("utf-8")
    with tempfile.TemporaryDirectory(prefix="engramdb-disc-") as td:
        root = Path(td)
        st_path = root / "model.safetensors"
        with open(st_path, "wb") as f:
            f.write(struct.pack("<Q", len(header_bytes)))
            f.write(header_bytes)
            f.write(payload)

        (root / "config.json").write_text(json.dumps({
            "architectures": ["Qwen4ExpForCausalLM"],
            "model_type": "qwen4_exp_text",
            "text_config": {
                "model_type": "qwen4_exp_text",
                "ple_layer_ids": [1],
                "ple_embed_dim": 2560,
                "ple_conv_kernel_size": 4,
                "ngram_size": 3,
                "ngram_vocab_size_base": 20_000_000,
                "split_ngram_parts": 128,
                "heads_per_ngram": 8,
                "make_ngram_vocab_size_divisible_by": 128,
            },
        }))
        (root / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {
                "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.weight_scale": "model.safetensors",
                "model.language_model.layers.1.ple.ple_embedding.layer_multipliers": "model.safetensors",
                "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight": "model.safetensors",
            }
        }))

        info = discover_ple(root)
        assert info is not None
        assert info["ngram_embedding_shard_count"] == 1
        assert info["layer_multipliers"] == multipliers
        assert info["rowid_multipliers"] == multipliers
        assert abs(float(info["weight_scale"]) - scale) < 1e-12
        assert load_ple_multipliers(root) == multipliers
        assert abs(load_ple_weight_scale(root) - scale) < 1e-12
    print("discover_ple metadata OK")


def test_official_loader_filter() -> None:
    from engramdb.official_loader import filter_ngram_shard_state_dict

    state = {
        "model.language_model.layers.1.ple.key_proj.weight": "keep",
        "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight": "drop",
        "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.weight": "drop",
        "model.language_model.layers.1.self_attn.q_proj.weight": "keep",
    }
    filtered = filter_ngram_shard_state_dict(state)
    assert "model.language_model.layers.1.ple.key_proj.weight" in filtered
    assert "model.language_model.layers.1.self_attn.q_proj.weight" in filtered
    assert len(filtered) == 2
    print("official_loader filter OK")



def test_official_loader_placeholder_patch() -> None:
    try:
        import torch
    except Exception as exc:
        print(f"official_loader placeholder patch skipped (no torch: {exc})")
        return

    import sys
    import types

    from engramdb.official_loader import patch_official_ngram_embedding_for_disk_load

    fake_mod = types.ModuleType("fake_qwen4_exp_smoke")
    fake_mod.nn = torch.nn
    sys.modules[fake_mod.__name__] = fake_mod

    class FakeNGramEmbedding(torch.nn.Module):
        def __init__(self, rows: int = 1_000_000, dim: int = 8):
            super().__init__()
            self.ngram_embedding = torch.nn.Embedding(rows, dim)

    FakeNGramEmbedding.__module__ = fake_mod.__name__
    fake_mod.FakeNGramEmbedding = FakeNGramEmbedding

    with patch_official_ngram_embedding_for_disk_load(
        embedding_class=FakeNGramEmbedding
    ):
        inside = FakeNGramEmbedding(1_000_000, 8)
        assert inside.ngram_embedding.weight.shape[0] == 1
        assert inside.ngram_embedding._requested_num_embeddings == 1_000_000

    outside = FakeNGramEmbedding(16, 8)
    assert outside.ngram_embedding.weight.shape[0] == 16
    print("official_loader placeholder patch OK")


def test_official_loader_sharded_load() -> None:
    try:
        import torch
        from safetensors.torch import save_file
    except Exception as exc:
        print(f"official_loader sharded load skipped (no torch/safetensors: {exc})")
        return

    import json as _json

    from engramdb.official_loader import load_official_checkpoint_without_ngram_shards

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Sequential()
            self.model.linear = torch.nn.Linear(3, 2)

    with tempfile.TemporaryDirectory(prefix="engramdb-official-load-") as td:
        root = Path(td)
        shard_a = {
            "model.linear.weight": torch.ones(2, 3),
            "model.linear.bias": torch.zeros(2),
        }
        shard_b = {
            "model.ple.ngram_embedding.shard_0.weight": torch.zeros(16, 8),
            "model.ple.ngram_embedding.weight_scale": torch.tensor([1.0]),
        }
        save_file(shard_a, root / "model-00001.safetensors")
        save_file(shard_b, root / "model-00002.safetensors")
        (root / "model.safetensors.index.json").write_text(_json.dumps({
            "weight_map": {
                "model.linear.weight": "model-00001.safetensors",
                "model.linear.bias": "model-00001.safetensors",
                "model.ple.ngram_embedding.shard_0.weight": "model-00002.safetensors",
                "model.ple.ngram_embedding.weight_scale": "model-00002.safetensors",
            }
        }))

        model = FakeModel()
        result = load_official_checkpoint_without_ngram_shards(
            model, root, strict=False
        )
        assert result.loaded_tensors == 2
        assert result.skipped_ngram_tensors == 2
        assert torch.all(model.model.linear.weight == 1.0)
        assert torch.all(model.model.linear.bias == 0.0)
    print("official_loader sharded load OK")

def test_rowids_for_seq() -> None:
    rows = engramdb.rowids_for_seq([1000, 99999, 42])
    assert len(rows) == 3
    assert all(len(r) == 16 for r in rows)
    assert rows[0][0] == 1876085

    # The pure-Python path must accept explicit multipliers and discovery info.
    multipliers = [23_703_573_157_769, 20_109_073_645_365, 8_052_911_324_071]
    rows_custom = engramdb.rowids_for_seq(
        [1000, 99999, 42], multipliers=multipliers
    )
    assert rows_custom == rows, "custom multipliers should match standard golden"
    rows_info = engramdb.rowids_for_seq(
        [1000, 99999, 42],
        info={
            "layer_multipliers": multipliers,
            "rowid_multipliers": multipliers,
        },
    )
    assert rows_info == rows, "info multipliers should match standard golden"
    print("rowids_for_seq OK:", rows[0][:4])


def test_store_pool() -> None:
    from engramdb import StorePool, ThreadLocalStore

    with tempfile.TemporaryDirectory(prefix="engramdb-pool-") as td:
        root = Path(td)
        _make_table(root, "pool", rows=8, width=4)
        directory = str(root / "pool")
        pool = StorePool(
            directory,
            shards=1,
            rows_per_shard=8,
            width=4,
            pool_size=2,
        )
        try:
            with pool as store:
                assert store.fetch([1, 3]) == bytes([1] * 4) + bytes([3] * 4)

            tls = ThreadLocalStore(pool)
            handle = tls.get()
            assert handle.fetch([0]) == bytes([0] * 4)
            tls.release_current()

            stats = pool.stats()
            assert stats["acquires"] == 2
            assert stats["releases"] == 2
            assert stats["borrowed"] == 0
            assert stats["waits"] >= 0
        finally:
            pool.close()
    print("StorePool OK")



def test_ple_memory() -> None:
    """Exercise the optional serving-layer PleMemory/PleSequence API."""
    # This test intentionally needs only Store, not torch.  Use a tiny custom
    # prime table so generated rowids stay inside the synthetic shard.
    import struct

    from engramdb.ple_memory import PleMemory, PleSequenceStore

    head_dim = 4
    num_heads = 2
    # 4 rows total; each row is one head of width 4 bytes.
    rows = [struct.pack("<HH", i, i + 1) for i in range(4)]
    with tempfile.TemporaryDirectory(prefix="engramdb-ple-memory-") as directory:
        with open(os.path.join(directory, "shard_000.bin"), "wb") as f:
            for row in rows:
                f.write(row)
        store = engramdb.Store(directory, 1, len(rows), head_dim)
        try:
            mem = PleMemory(
                store=store,
                head_dim=head_dim,
                num_heads=num_heads,
                ngram_size=2,
                heads_per_ngram=2,
                prime_sizes=[2, 2],
                offsets=[0, 2],
                eos=0,
            )
            assert mem.source == "store"
            assert mem.record_bytes == head_dim * num_heads

            rows_for_tokens = mem.rowids_for_tokens([1, 2])
            assert len(rows_for_tokens) == 2
            assert all(len(row) == num_heads for row in rows_for_tokens)

            raw = mem.fetch_raw(rows_for_tokens)
            assert len(raw) == 2 * mem.record_bytes

            seq = mem.new_sequence()
            step = seq.feed([1, 2])
            assert step.tokens == [1, 2]
            assert len(step.raw) == 2 * mem.record_bytes
            assert seq.history == [2]
            assert seq.current_rowids() == rows_for_tokens
            assert seq.current_raw() == raw

            step2 = seq.feed([3])
            assert step2.tokens == [3]
            assert seq.history == [3]
            assert seq.length == 3
            seq.reset()
            assert seq.length == 0
            assert seq.history == [0]

            store = PleSequenceStore(mem, max_sequences=2)
            store.feed("a", [1, 2])
            store.feed("b", [3])
            assert len(store) == 2
            assert store.get("a").history == [2]
            assert store.current_raw("b") == mem.fetch_raw(mem.rowids_for_tokens([3], [0]))
            store.remove("b")
            assert len(store) == 1
            store.clear()
            assert len(store) == 0

            # Optional torch tensor path (skip silently when unavailable).
            try:
                import torch
            except Exception:
                torch = None
            if torch is not None:
                t = mem.fetch_tensor(rows_for_tokens)
                assert tuple(t.shape) == (2, num_heads, head_dim)
                assert t.dtype == torch.float32
                print("PleMemory tensor OK")
            print("PleMemory OK")
        finally:
            store.close()


def test_bundle_and_target_reader() -> None:
    """Exercise bundle manifest resolution and generic reader registry."""
    import struct

    import engramdb.bundle as bundle_mod
    import engramdb.target_reader as reader_mod

    with tempfile.TemporaryDirectory(prefix="engramdb-bundle-") as td:
        root = Path(td)
        rows_dir = root / "rows"
        rows_dir.mkdir()
        # 4 rows, each 4 bytes; enough for a tiny Store-I bundle.
        with open(rows_dir / "shard_000.bin", "wb") as f:
            for i in range(4):
                f.write(struct.pack("<HH", i, i + 1))

        manifest_data = {
            "schema": "engramdb-bundle-v1",
            "id": "test-bundle",
            "memory": {
                "type": "store",
                "store": {
                    "path": "rows",
                    "shards": 1,
                    "rows_per_shard": 4,
                    "width": 4,
                },
            },
            "ple": {
                "ple_embed_dim": 8,
                "num_heads": 2,
                "head_dim": 4,
                "ngram_size": 2,
                "heads_per_ngram": 2,
                "prime_sizes": [2, 2],
                "offsets": [0, 2],
                "eos": 0,
            },
            "readers": [
                {
                    "name": "dummy",
                    "version": "1",
                    "path": "reader.bin",
                    "options": {"flag": True},
                }
            ],
        }
        manifest_path = root / "bundle.json"
        manifest_path.write_text(json.dumps(manifest_data), encoding="utf-8")

        bm = bundle_mod.BundleManifest.load(manifest_path)
        assert bm.validate() == [], bm.validate()
        resolved = bm.resolved()
        assert resolved["memory"]["store"]["path"] == str(rows_dir)
        assert resolved["readers"][0]["path"] == str(root / "reader.bin")

        mem = bm.open_memory()
        try:
            assert mem.source == "store"
            assert mem.num_heads == 2
            assert mem.record_bytes == 8
        finally:
            mem.store.close()

        reg = reader_mod.TargetReaderRegistry()

        @reg.register("dummy", version="1")
        def build_dummy(**kwargs: object) -> object:
            return kwargs

        reader = reg.create_from_manifest(bm)
        assert reader["flag"] is True
        assert reader["path"] == str(root / "reader.bin")
        assert reg.available() == [{"name": "dummy", "version": "1"}]
        print("Bundle + TargetReader OK")


def test_engine_adapter() -> None:
    """Exercise the generic PyTorch adapter and target-reader hook."""
    try:
        import torch
        from torch import nn
    except Exception:
        print("Engine adapter skipped: torch not available")
        return

    import struct

    from engramdb.adapter import (
        PleMemoryAdapter,
        install_target_reader_hook,
    )
    from engramdb.ple_memory import PleMemory

    head_dim = 4
    num_heads = 2
    rows = [struct.pack("<HH", i, i + 1) for i in range(4)]
    with tempfile.TemporaryDirectory(prefix="engramdb-adapter-") as directory:
        with open(os.path.join(directory, "shard_000.bin"), "wb") as f:
            for row in rows:
                f.write(row)
        store = engramdb.Store(directory, 1, len(rows), head_dim)
        try:
            mem = PleMemory(
                store=store,
                head_dim=head_dim,
                num_heads=num_heads,
                ngram_size=2,
                heads_per_ngram=2,
                prime_sizes=[2, 2],
                offsets=[0, 2],
                eos=0,
            )
            adapter = PleMemoryAdapter(mem, keep_steps=4)
            out = adapter(torch.tensor([[1, 2], [3, 1]]), seq_ids=[10, 20])
            assert tuple(out.shape) == (2, 2, num_heads, head_dim)
            assert len(adapter.sequences) == 2
            adapter.reset()
            assert len(adapter.sequences) == 0

            model = nn.Identity()
            reader_calls = {"n": 0}

            class DummyReader:
                def on_forward(self, module: object, args: object, output: object) -> object:
                    reader_calls["n"] += 1
                    return output

            hook = install_target_reader_hook(model, DummyReader(), mode="post")
            out = model(torch.tensor([1.0, 2.0]))
            assert reader_calls["n"] == 1
            assert tuple(out.shape) == (2,)
            hook.remove()
            model(torch.tensor([1.0]))
            assert reader_calls["n"] == 1
            print("Engine adapter OK")
        finally:
            store.close()


def main() -> None:
    from importlib.metadata import version as _dist_version

    try:
        dist_version = _dist_version("engramdb-python")
    except Exception:
        dist_version = engramdb.__version__
    assert engramdb.__version__ == dist_version, (
        f"module {engramdb.__version__} != dist {dist_version}"
    )
    # Importing every public integration surface catches missing/renamed symbols.
    from engramdb.vllm import PleDiskGather  # noqa: F401
    from engramdb import sglang  # noqa: F401
    try:
        from engramdb import vllm_plugin  # noqa: F401
        print("vllm_plugin import OK")
    except Exception as exc:
        print(f"vllm_plugin skipped ({exc})")
    try:
        from engramdb import integrations  # noqa: F401
        print("integrations import OK")
    except Exception as exc:  # optional torch/engram-peft dependency
        print(f"integrations skipped ({exc})")
    try:
        from engramdb import adapter  # noqa: F401
        print("adapter import OK")
    except Exception as exc:
        print(f"adapter skipped ({exc})")

    test_page_reader()
    test_slot_index()
    test_disk_slot_index()
    test_store_and_vllm_gather()
    test_store_concurrent_fetch()
    test_safetensors_i64_reader()
    test_discover_ple_metadata()
    test_official_loader_filter()
    test_official_loader_placeholder_patch()
    test_official_loader_sharded_load()
    test_rowids_for_seq()
    test_store_pool()
    test_ple_memory()
    test_bundle_and_target_reader()
    test_engine_adapter()
    test_database_arrow_server()
    test_disk_ple_lru()
    test_prefetch_lru()
    test_fetch_e_t_tensor()
    test_disk_ple_nocache_fastpath()
    print("python wheel smoke OK")


if __name__ == "__main__":
    main()
