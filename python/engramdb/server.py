"""Minimal EngramDB service prototype.

This is a small TCP/JSON service that exposes basic multi-table reads to remote
clients.  It is intentionally storage-only: it does not implement scheduling,
Arrow IPC yet, or engine integration.  The protocol is newline-delimited JSON.

Implemented commands:
  {"cmd": "ping"}
  {"cmd": "list_tables"}
  {"cmd": "fetch", "table": "...", "rowids": [...],
   "shards": n, "rows_per_shard": n, "width": n}
  {"cmd": "view_read", "path": "/...", "index": i}
"""

from __future__ import annotations

import json
import socketserver
from typing import Any

from . import Store, View
from .tables import Database


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


class EngramDBServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, database: Database, host: str = "127.0.0.1", port: int = 8765) -> None:
        super().__init__((host, port), _Handler)
        self.database = database
