"""Minimal EngramDB service prototype.

The service is storage-only and intentionally does not implement scheduling,
engine integration, or a full query protocol.

Two wire modes are provided:

* ``EngramDBServer`` -- newline-delimited JSON (backward-compatible prototype).
* ``EngramDBBinaryServer`` -- length-prefixed binary frames.

Binary frame layout
-------------------

Request::

    4-byte big-endian payload length
    UTF-8 JSON request

Response::

    4-byte big-endian frame length
    1-byte kind + payload

Kinds:

* ``0`` -- JSON response body
* ``1`` -- raw bytes (e.g. Store fetch or View slot)
* ``2`` -- Arrow IPC stream bytes

Implemented commands:

* ``ping``
* ``list_tables``
* ``fetch`` (JSON mode)
* ``fetch_raw`` (binary mode)
* ``fetch_arrow`` (Arrow IPC bytes)
* ``view_read`` (binary mode returns raw slot bytes)
"""

from __future__ import annotations

import json
import socketserver
from typing import Any

from . import Store, View
from .arrow_utils import store_fetch_arrow, table_to_ipc_bytes
from .tables import Database

KIND_JSON = 0
KIND_RAW = 1
KIND_ARROW = 2


def _read_exact(stream: Any, n: int) -> bytes:
    """Read exactly ``n`` bytes from a socket stream."""
    chunks: list[bytes] = []
    remaining = n
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _json_response(obj: dict[str, Any]) -> tuple[int, bytes]:
    return KIND_JSON, json.dumps(obj).encode("utf-8")


def _fetch_raw(server: "EngramDBServer", req: dict[str, Any]) -> bytes:
    table = req["table"]
    rowids = [int(x) for x in req.get("rowids", [])]
    return server.database.fetch(
        table,
        rowids,
        shards=int(req["shards"]),
        rows_per_shard=int(req["rows_per_shard"]),
        width=int(req["width"]),
    )


def _binary_dispatch(server: "EngramDBServer", req: dict[str, Any]) -> tuple[int, bytes]:
    """Dispatch a binary-protocol request to a ``(kind, payload)`` response."""
    cmd = req.get("cmd")
    if cmd == "ping":
        return _json_response({"ok": True, "pong": True})
    if cmd == "list_tables":
        return _json_response({"ok": True, "tables": server.database.list_tables()})
    if cmd == "fetch_raw":
        return KIND_RAW, _fetch_raw(server, req)
    if cmd == "fetch_arrow":
        table = req["table"]
        rowids = [int(x) for x in req.get("rowids", [])]
        store = Store(
            str(server.database.root / table),
            int(req["shards"]),
            int(req["rows_per_shard"]),
            int(req["width"]),
        )
        try:
            arrow_table = store_fetch_arrow(store, rowids)
            return KIND_ARROW, table_to_ipc_bytes(arrow_table)
        finally:
            store.close()
    if cmd == "view_read":
        view = View(req["path"])
        try:
            return KIND_RAW, view.read_record(int(req["index"]))
        finally:
            view.close()
    raise ValueError(f"unknown command: {cmd!r}")


class _Handler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        server: "EngramDBServer" = self.server  # type: ignore[assignment]
        for line in self.rfile:
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
                resp = self._dispatch(server, req)
            except Exception as exc:  # keep the server alive on bad requests
                resp = {"ok": False, "error": repr(exc)}
            self.wfile.write((json.dumps(resp) + "\n").encode("utf-8"))
            self.wfile.flush()

    def _dispatch(self, server: "EngramDBServer", req: dict[str, Any]) -> dict[str, Any]:
        cmd = req.get("cmd")
        if cmd == "ping":
            return {"ok": True, "pong": True}
        if cmd == "list_tables":
            return {"ok": True, "tables": server.database.list_tables()}
        if cmd == "fetch":
            table = req["table"]
            rowids = [int(x) for x in req.get("rowids", [])]
            raw = server.database.fetch(
                table,
                rowids,
                shards=int(req["shards"]),
                rows_per_shard=int(req["rows_per_shard"]),
                width=int(req["width"]),
            )
            return {
                "ok": True,
                "rowids": rowids,
                "width": int(req["width"]),
                "raw_base64": __import__("base64").b64encode(raw).decode("ascii"),
            }
        if cmd == "fetch_arrow":
            table = req["table"]
            rowids = [int(x) for x in req.get("rowids", [])]
            store = Store(
                str(server.database.root / table),
                int(req["shards"]),
                int(req["rows_per_shard"]),
                int(req["width"]),
            )
            try:
                arrow_table = store_fetch_arrow(store, rowids)
                ipc = table_to_ipc_bytes(arrow_table)
            finally:
                store.close()
            return {
                "ok": True,
                "num_rows": len(rowids),
                "ipc_base64": __import__("base64").b64encode(ipc).decode("ascii"),
            }
        if cmd == "view_read":
            view = View(req["path"])
            try:
                data = view.read_record(int(req["index"]))
                return {
                    "ok": True,
                    "index": int(req["index"]),
                    "slot_base64": __import__("base64").b64encode(data).decode("ascii"),
                }
            finally:
                view.close()
        raise ValueError(f"unknown command: {cmd!r}")


class _BinaryHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        server: "EngramDBBinaryServer" = self.server  # type: ignore[assignment]
        while True:
            header = _read_exact(self.rfile, 4)
            if len(header) < 4:
                return
            body_len = int.from_bytes(header, "big")
            body = _read_exact(self.rfile, body_len)
            try:
                req = json.loads(body.decode("utf-8"))
                kind, payload = _binary_dispatch(server, req)
            except Exception as exc:  # keep the server alive on bad requests
                kind, payload = _json_response({"ok": False, "error": repr(exc)})
            frame = bytes([kind]) + payload
            self.wfile.write(len(frame).to_bytes(4, "big"))
            self.wfile.write(frame)
            self.wfile.flush()


class EngramDBBinaryServer(socketserver.ThreadingTCPServer):
    """Length-prefixed binary-protocol EngramDB service.

    The first response byte selects the payload kind (JSON, raw bytes, or Arrow
    IPC).  This avoids base64-wrapping large reads and gives clients a direct
    Arrow IPC stream.
    """

    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, database: Database, host: str = "127.0.0.1", port: int = 8765) -> None:
        super().__init__((host, port), _BinaryHandler)
        self.database = database


class EngramDBServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, database: Database, host: str = "127.0.0.1", port: int = 8765) -> None:
        super().__init__((host, port), _Handler)
        self.database = database
