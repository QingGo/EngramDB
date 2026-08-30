"""EngramDB: disk-first storage engine for Engram/PLE n-gram memory tables.

This package is the Python face of the Rust core.  It loads the small C ABI
cdylib built by `crates/engramdb-python` and exposes a ctypes-based `Store`
and `View` API.  It is intentionally minimal: the goal is to let
Python/PyTorch code (especially `engram-peft`) read fixed-size rows/records
from an EngramDB layout without a full PyO3/maturin stack.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

__version__ = "0.1.0"


def _find_library() -> str:
    root = Path(__file__).resolve().parents[2]
    candidates = [
        root / "target" / "release" / "libengramdb_c.dylib",
        root / "target" / "debug" / "libengramdb_c.dylib",
        root / "target" / "release" / "libengramdb_c.so",
        root / "target" / "debug" / "libengramdb_c.so",
        root / "target" / "release" / "libengramdb_c.dll",
        root / "target" / "debug" / "libengramdb_c.dll",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    raise ImportError("EngramDB native library not found; run cargo build -p engramdb-python --release")


_lib = ctypes.CDLL(_find_library())

_lib.engramdb_store_open.restype = ctypes.c_void_p
_lib.engramdb_store_open.argtypes = [
    ctypes.c_char_p,
    ctypes.c_uint64,
    ctypes.c_uint64,
    ctypes.c_uint64,
]

_lib.engramdb_store_fetch.restype = ctypes.c_int
_lib.engramdb_store_fetch.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_uint64),
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_uint8),
    ctypes.c_size_t,
]

_lib.engramdb_store_width.restype = ctypes.c_uint64
_lib.engramdb_store_width.argtypes = [ctypes.c_void_p]

_lib.engramdb_store_close.restype = None
_lib.engramdb_store_close.argtypes = [ctypes.c_void_p]

_lib.engramdb_view_open.restype = ctypes.c_void_p
_lib.engramdb_view_open.argtypes = [ctypes.c_char_p]

_lib.engramdb_view_read_record.restype = ctypes.c_int
_lib.engramdb_view_read_record.argtypes = [
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_uint8),
    ctypes.c_size_t,
]

_lib.engramdb_view_len.restype = ctypes.c_size_t
_lib.engramdb_view_len.argtypes = [ctypes.c_void_p]

_lib.engramdb_view_slot_bytes.restype = ctypes.c_uint64
_lib.engramdb_view_slot_bytes.argtypes = [ctypes.c_void_p]

_lib.engramdb_view_close.restype = None
_lib.engramdb_view_close.argtypes = [ctypes.c_void_p]


class Store:
    """Open a Store-I shard directory as a flat fixed-size row store."""

    def __init__(self, directory: str, shards: int, rows_per_shard: int, width: int):
        handle = _lib.engramdb_store_open(
            directory.encode("utf-8"),
            ctypes.c_uint64(shards),
            ctypes.c_uint64(rows_per_shard),
            ctypes.c_uint64(width),
        )
        if not handle:
            raise OSError(f"failed to open EngramDB store: {directory}")
        self._handle = handle
        self._closed = False

    @property
    def width(self) -> int:
        return int(_lib.engramdb_store_width(self._handle))

    def fetch(self, rowids: list[int]) -> bytes:
        if self._closed:
            raise ValueError("store is closed")
        if not rowids:
            return b""
        n = len(rowids)
        arr = (ctypes.c_uint64 * n)(*rowids)
        width = self.width
        out = (ctypes.c_uint8 * (n * width))()
        rc = _lib.engramdb_store_fetch(
            self._handle,
            arr,
            n,
            out,
            n * width,
        )
        if rc != 0:
            raise OSError(f"engramdb_store_fetch failed with code {rc}")
        return bytes(out)

    def fetch_one(self, rowid: int) -> bytes:
        return self.fetch([rowid])

    def close(self) -> None:
        if not self._closed:
            _lib.engramdb_store_close(self._handle)
            self._closed = True

    def __del__(self):  # pragma: no cover - best-effort cleanup
        try:
            self.close()
        except Exception:
            pass


class View:
    """Open a Store-P materialized view and read fixed-size records."""

    def __init__(self, path: str):
        handle = _lib.engramdb_view_open(path.encode("utf-8"))
        if not handle:
            raise OSError(f"failed to open EngramDB view: {path}")
        self._handle = handle
        self._closed = False

    def __len__(self) -> int:
        return int(_lib.engramdb_view_len(self._handle))

    @property
    def slot_bytes(self) -> int:
        return int(_lib.engramdb_view_slot_bytes(self._handle))

    def read_record(self, index: int) -> bytes:
        if self._closed:
            raise ValueError("view is closed")
        size = self.slot_bytes
        out = (ctypes.c_uint8 * size)()
        rc = _lib.engramdb_view_read_record(self._handle, index, out, size)
        if rc != 0:
            raise OSError(f"engramdb_view_read_record failed with code {rc}")
        return bytes(out)

    def close(self) -> None:
        if not self._closed:
            _lib.engramdb_view_close(self._handle)
            self._closed = True

    def __del__(self):  # pragma: no cover - best-effort cleanup
        try:
            self.close()
        except Exception:
            pass


def read_keys(path: str) -> list[int]:
    """Read a rowid text file into a Python list (one u64 per line)."""
    result: list[int] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                result.append(int(line))
    return result


def __repr__() -> str:  # pragma: no cover
    return f"<engramdb {__version__} ctypes bindings>"
