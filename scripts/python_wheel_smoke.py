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

    row_width = 4
    rows = [bytes([i] * row_width) for i in range(8)]
    with tempfile.TemporaryDirectory(prefix="engramdb-lru-") as directory:
        with open(os.path.join(directory, "shard_000.bin"), "wb") as f:
            for row in rows:
                f.write(row)
        store = engramdb.Store(directory, 1, len(rows), row_width)
        try:
            emb = DiskPleEmbedding(
                store,
                num_embeddings=len(rows),
                embedding_dim=row_width,
                dtype=torch.float32,
                cache_size=4,
            )
            # A single 3-token lookup exercises miss + LRU fill + cache hit.
            indices = torch.tensor([2, 0, 2, 1, 2])
            out = emb(indices)
            assert tuple(out.shape) == (5, row_width)
            assert out.dtype == torch.float32
            assert len(emb._cache) <= emb.cache_size
            # The cache should contain a subset of the accessed rows.
            assert set(emb._cache.keys()).issubset({0, 1, 2})
            print("DiskPleEmbedding LRU OK:", tuple(out.shape), "cache", len(emb._cache))
        finally:
            store.close()


def test_rowids_for_seq() -> None:
    rows = engramdb.rowids_for_seq([1000, 99999, 42])
    assert len(rows) == 3
    assert all(len(r) == 16 for r in rows)
    assert rows[0][0] == 1876085
    print("rowids_for_seq OK:", rows[0][:4])


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

    test_page_reader()
    test_store_and_vllm_gather()
    test_rowids_for_seq()
    test_database_arrow_server()
    test_disk_ple_lru()
    print("python wheel smoke OK")


if __name__ == "__main__":
    main()
