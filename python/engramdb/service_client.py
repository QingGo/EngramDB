"""Binary-protocol client for the EngramDB service.

This client speaks the length-prefixed binary protocol implemented by
``EngramDBBinaryServer``.  It keeps the common operations simple and returns
raw bytes directly for ``fetch_raw`` / ``view_read`` and an Arrow IPC stream for
``fetch_arrow``.
"""

from __future__ import annotations

import json
import socket
from typing import Any

from .server import KIND_ARROW, KIND_JSON, KIND_RAW


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    chunks: list[bytes] = []
    remaining = n
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("connection closed while reading response")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


class EngramDBClient:
    """Small synchronous client for ``EngramDBBinaryServer``."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8765, timeout: float = 10.0) -> None:
        self._sock = socket.create_connection((host, port), timeout=timeout)

    def close(self) -> None:
        self._sock.close()

    def __enter__(self) -> "EngramDBClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def request(self, req: dict[str, Any]) -> tuple[int, bytes]:
        """Send one request and return ``(kind, payload)``."""
        body = json.dumps(req).encode("utf-8")
        self._sock.sendall(len(body).to_bytes(4, "big") + body)
        header = _recv_exact(self._sock, 4)
        frame_len = int.from_bytes(header, "big")
        frame = _recv_exact(self._sock, frame_len)
        if not frame:
            raise ConnectionError("empty binary response")
        return frame[0], frame[1:]

    def _json(self, kind: int, payload: bytes) -> dict[str, Any]:
        if kind != KIND_JSON:
            raise RuntimeError(f"expected JSON response, got kind={kind}")
        return json.loads(payload.decode("utf-8"))

    def ping(self) -> bool:
        resp = self._json(*self.request({"cmd": "ping"}))
        return bool(resp.get("ok") and resp.get("pong"))

    def list_tables(self) -> list[str]:
        resp = self._json(*self.request({"cmd": "list_tables"}))
        return list(resp.get("tables", []))

    def fetch_raw(
        self,
        table: str,
        rowids: list[int],
        shards: int,
        rows_per_shard: int,
        width: int,
    ) -> bytes:
        kind, payload = self.request({
            "cmd": "fetch_raw",
            "table": table,
            "rowids": rowids,
            "shards": shards,
            "rows_per_shard": rows_per_shard,
            "width": width,
        })
        if kind != KIND_RAW:
            raise RuntimeError(f"expected raw response, got kind={kind}")
        return payload

    def fetch_arrow(
        self,
        table: str,
        rowids: list[int],
        shards: int,
        rows_per_shard: int,
        width: int,
    ) -> bytes:
        kind, payload = self.request({
            "cmd": "fetch_arrow",
            "table": table,
            "rowids": rowids,
            "shards": shards,
            "rows_per_shard": rows_per_shard,
            "width": width,
        })
        if kind != KIND_ARROW:
            raise RuntimeError(f"expected Arrow response, got kind={kind}")
        return payload

    def view_read(self, path: str, index: int) -> bytes:
        kind, payload = self.request({"cmd": "view_read", "path": path, "index": index})
        if kind != KIND_RAW:
            raise RuntimeError(f"expected raw response, got kind={kind}")
        return payload
