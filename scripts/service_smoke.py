#!/usr/bin/env python3
"""Smoke-test multi-table Database, optional Arrow helpers, and TCP service."""
import base64
import json
import socket
import tempfile
import threading
from pathlib import Path

from engramdb import Database
from engramdb.server import EngramDBServer


def make_table(root: Path, name: str, rows: int = 4, width: int = 8) -> None:
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "shard_000.bin", "wb") as f:
        for i in range(rows):
            f.write(bytes([i % 256]) * width)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="engramdb-service-") as td:
        root = Path(td)
        make_table(root, "alpha")
        make_table(root, "beta")

        db = Database(root)
        assert db.list_tables() == ["alpha", "beta"], db.list_tables()

        raw = db.fetch("alpha", [1, 3], shards=1, rows_per_shard=4, width=8)
        assert len(raw) == 16
        print("Database OK:", db.list_tables(), raw[:8])

        # Optional Arrow path
        try:
            from engramdb.arrow_utils import store_fetch_arrow, table_to_ipc_bytes
            table = store_fetch_arrow(db.open_store("alpha", 1, 4, 8), [0, 2])
            ipc = table_to_ipc_bytes(table)
            print("Arrow OK:", table.num_rows, table.column_names, "ipc_bytes", len(ipc))
            assert len(ipc) > 0
        except ImportError as exc:
            print("Arrow skipped:", exc)

        # TCP service
        server = EngramDBServer(db, host="127.0.0.1", port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        host, port = server.server_address

        def call(req: dict) -> dict:
            with socket.create_connection((host, port), timeout=5) as sock:
                sock.sendall((json.dumps(req) + "\n").encode("utf-8"))
                data = sock.makefile("rb").readline()
            return json.loads(data)

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
        assert base64.b64decode(fet["raw_base64"]) == db.fetch(
            "alpha", [0, 2], shards=1, rows_per_shard=4, width=8
        )
        try:
            import pyarrow as pa
            arrow_resp = call({
                "cmd": "fetch_arrow",
                "table": "alpha",
                "rowids": [0, 2],
                "shards": 1,
                "rows_per_shard": 4,
                "width": 8,
            })
            assert arrow_resp["ok"]
            ipc_bytes = base64.b64decode(arrow_resp["ipc_base64"])
            with pa.ipc.open_stream(ipc_bytes) as reader:
                arrow_table = reader.read_all()
            assert arrow_table.num_rows == 2
            print("Server Arrow OK:", arrow_table.num_rows, arrow_table.column_names)
        except ImportError:
            print("Server Arrow skipped: pyarrow not available")

        print("Server OK:", ping, tables, len(base64.b64decode(fet["raw_base64"])))

        server.shutdown()
        server.server_close()
        db.close()
        print("SERVICE_SMOKE_OK")


if __name__ == "__main__":
    main()
