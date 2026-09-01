
"""EngramDB: disk-first storage engine for Engram/PLE n-gram memory tables.

The package prefers the native PyO3 extension (`_engramdb`) when present.
If that is unavailable, it falls back to the ctypes C-ABI bridge; if neither
is available, importing the module still works but Store/View will raise on use.
"""

from __future__ import annotations

from typing import Any

__version__ = "0.2.12"

_USING_PYO3 = False
_USING_CTYPES = False

try:
    from ._engramdb import Store, View, read_keys
    try:
        from ._engramdb import PageReader
    except ImportError:
        PageReader = None  # type: ignore[assignment]
    try:
        from ._engramdb import IoUringPageReader
    except ImportError:
        IoUringPageReader = None  # type: ignore[assignment]
except ImportError:
    pass
else:
    _USING_PYO3 = True


if not _USING_PYO3:
    try:
        import ctypes
        from pathlib import Path
    except ImportError:
        ctypes = None  # type: ignore[assignment]
        Path = None  # type: ignore[assignment]

    if ctypes is not None:
        def _find_library() -> str | None:
            root = Path(__file__).resolve().parents[2] if Path else None
            if root is None:
                return None
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
            return None

        _lib_path = _find_library()
        if _lib_path is not None:
            _lib = ctypes.CDLL(_lib_path)

            if hasattr(_lib, "engramdb_abi_version"):
                _lib.engramdb_abi_version.restype = ctypes.c_uint32
                _lib.engramdb_rowids_for_seq.restype = ctypes.c_int
                _lib.engramdb_rowids_for_seq.argtypes = [
                    ctypes.POINTER(ctypes.c_uint32),
                    ctypes.c_size_t,
                    ctypes.POINTER(ctypes.c_uint64),
                    ctypes.c_size_t,
                    ctypes.c_uint32,
                ]

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
                def __init__(self, directory, shards, rows_per_shard, width):
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
                def width(self):
                    return int(_lib.engramdb_store_width(self._handle))

                def fetch(self, rowids):
                    if self._closed:
                        raise ValueError("store is closed")
                    if not rowids:
                        return b""
                    n = len(rowids)
                    arr = (ctypes.c_uint64 * n)(*rowids)
                    width = self.width
                    out = (ctypes.c_uint8 * (n * width))()
                    rc = _lib.engramdb_store_fetch(self._handle, arr, n, out, n * width)
                    if rc != 0:
                        raise OSError(f"engramdb_store_fetch failed with code {rc}")
                    return bytes(out)

                def fetch_one(self, rowid):
                    return self.fetch([rowid])

                def close(self):
                    if not self._closed:
                        _lib.engramdb_store_close(self._handle)
                        self._closed = True

                def __del__(self):
                    try:
                        self.close()
                    except Exception:
                        pass

            class View:
                def __init__(self, path):
                    handle = _lib.engramdb_view_open(path.encode("utf-8"))
                    if not handle:
                        raise OSError(f"failed to open EngramDB view: {path}")
                    self._handle = handle
                    self._closed = False

                def __len__(self):
                    return int(_lib.engramdb_view_len(self._handle))

                @property
                def slot_bytes(self):
                    return int(_lib.engramdb_view_slot_bytes(self._handle))

                def read_record(self, index):
                    if self._closed:
                        raise ValueError("view is closed")
                    size = self.slot_bytes
                    out = (ctypes.c_uint8 * size)()
                    rc = _lib.engramdb_view_read_record(self._handle, index, out, size)
                    if rc != 0:
                        raise OSError(f"engramdb_view_read_record failed with code {rc}")
                    return bytes(out)

                def close(self):
                    if not self._closed:
                        _lib.engramdb_view_close(self._handle)
                        self._closed = True

                def __del__(self):
                    try:
                        self.close()
                    except Exception:
                        pass

            def read_keys(path):
                result = []
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            result.append(int(line))
                return result

            _USING_CTYPES = True


from .tables import Database
from .server import EngramDBServer, EngramDBBinaryServer
from .service_client import EngramDBClient
from .ple_discovery import (
    discover_ple,
    load_ple_multipliers,
    load_ple_weight_scale,
)
from .vllm import fetch_e_t_tensor
from .pool import StorePool, ThreadLocalStore
try:  # SlotIndex/DiskSlotIndex are pure-Python and need numpy; keep optional for light installs.
    from .slot_index import SlotIndex
    from .disk_slot_index import DiskSlotIndex
except ImportError:  # pragma: no cover - depends on environment
    SlotIndex = None  # type: ignore[assignment,misc]
    DiskSlotIndex = None  # type: ignore[assignment,misc]

PLE_QWEN_V1 = 1
ENG_DEEPSEEK_V1 = 2


def abi_version() -> int:
    """Return the EngramDB C ABI version implemented by this package."""
    if _USING_PYO3 and hasattr(_engramdb, "abi_version"):
        return int(_engramdb.abi_version())
    if _USING_CTYPES and hasattr(_lib, "engramdb_abi_version"):
        return int(_lib.engramdb_abi_version())
    return 1


def rowids_for_seq(
    tokens: Any,
    ple_spec: int = PLE_QWEN_V1,
    multipliers: list[int] | None = None,
    info: dict[str, Any] | None = None,
) -> list[list[int]]:
    """Return PLE/Engram rowids for a token sequence.

    The returned shape is ``[len(tokens), 16]`` for ``PLE_QWEN_V1``.  The
    implementation prefers the native Rust path (PyO3 or the C ABI) and falls
    back to the pure-Python reference when no native binding is installed.

    ``multipliers`` may be supplied directly, or taken from ``info`` as returned
    by :func:`engramdb.ple_discovery.discover_ple` (keys ``layer_multipliers``,
    ``rowid_multipliers``, or ``multipliers``).  When a non-default multiplier
    source is provided, the pure-Python path is used because the current C
    ABI/PyO3 rowid entry points implement only the standard Qwen PLE constants.
    """
    if ple_spec != PLE_QWEN_V1:
        raise NotImplementedError(
            f"ple_spec {ple_spec} is not implemented (only PLE_QWEN_V1=1)"
        )
    if hasattr(tokens, "tolist"):
        tokens = tokens.tolist()
    tok = [int(x) for x in tokens]

    info_multipliers: list[int] | None = None
    if info is not None:
        for key in ("layer_multipliers", "rowid_multipliers", "multipliers"):
            val = info.get(key)
            if val is not None:
                info_multipliers = [int(x) for x in val]
                break

    custom_multipliers = multipliers is not None or info_multipliers is not None
    if not custom_multipliers:
        if _USING_PYO3 and hasattr(_engramdb, "rowids_for_seq"):
            return [list(r) for r in _engramdb.rowids_for_seq(tok, ple_spec)]
        if _USING_CTYPES and hasattr(_lib, "engramdb_rowids_for_seq"):
            n = len(tok)
            if n == 0:
                return []
            arr = (ctypes.c_uint32 * n)(*tok)
            out = (ctypes.c_uint64 * (n * 16))()
            rc = _lib.engramdb_rowids_for_seq(arr, n, out, n * 16, ple_spec)
            if rc != 0:
                raise OSError(f"engramdb_rowids_for_seq failed with code {rc}")
            return [list(out[i * 16:(i + 1) * 16]) for i in range(n)]

    from .ple_adapter import ple_rowids

    if multipliers is not None:
        effective = [int(x) for x in multipliers]
    elif info_multipliers is not None:
        effective = info_multipliers
    else:
        effective = [23_703_573_157_769, 20_109_073_645_365, 8_052_911_324_071]
    return ple_rowids(tok, effective)


def rowids_for_seq_with_history(
    history: list[int],
    tokens: list[int],
    ple_spec: int = PLE_QWEN_V1,
) -> list[list[int]]:
    """Return PLE rowids for a streamed sequence with explicit n-gram history.

    This is the native fast path for sequential decode: ``history`` is the
    already-known previous context (usually ``ngram_size - 1`` tokens), and
    ``tokens`` are the current step's input ids.  Falls back to the pure-Python
    reference when the native binding is unavailable.
    """
    if ple_spec != PLE_QWEN_V1:
        raise NotImplementedError(
            f"ple_spec {ple_spec} is not implemented (only PLE_QWEN_V1=1)"
        )
    hist = [int(x) for x in history]
    tok = [int(x) for x in tokens]
    if _USING_PYO3 and hasattr(_engramdb, "rowids_for_seq_with_history"):
        return [list(r) for r in _engramdb.rowids_for_seq_with_history(hist, tok, ple_spec)]
    from .ple_adapter import ple_rowids

    return ple_rowids(tok, [23_703_573_157_769, 20_109_073_645_365, 8_052_911_324_071], history=hist)


def __getattr__(name: str):
    if name == "DiskPleNGramEmbedding":
        from .ple_adapter import DiskPleNGramEmbedding
        return DiskPleNGramEmbedding
    if name in (
        "PleMemory",
        "PleSequence",
        "PleSequenceStore",
        "PleStep",
        "ple_memory_from_discovery",
    ):
        from .ple_memory import (
            PleMemory,
            PleSequence,
            PleSequenceStore,
            PleStep,
            ple_memory_from_discovery,
        )
        return {
            "PleMemory": PleMemory,
            "PleSequence": PleSequence,
            "PleSequenceStore": PleSequenceStore,
            "PleStep": PleStep,
            "ple_memory_from_discovery": ple_memory_from_discovery,
        }[name]
    if name in ("BundleManifest", "bundle_manifest_from_path"):
        from .bundle import BundleManifest, bundle_manifest_from_path

        return {
            "BundleManifest": BundleManifest,
            "bundle_manifest_from_path": bundle_manifest_from_path,
        }[name]
    if name in ("TargetReaderRegistry", "ReaderSpec"):
        from .target_reader import ReaderSpec, TargetReaderRegistry

        return {
            "TargetReaderRegistry": TargetReaderRegistry,
            "ReaderSpec": ReaderSpec,
        }[name]
    if name in (
        "PleMemoryAdapter",
        "TargetReaderHook",
        "install_target_reader_hook",
        "install_bundle_adapter",
        "install_vllm_target_reader",
        "install_sglang_target_reader",
    ):
        from .adapter import (
            PleMemoryAdapter,
            TargetReaderHook,
            install_bundle_adapter,
            install_sglang_target_reader,
            install_target_reader_hook,
            install_vllm_target_reader,
        )
        return {
            "PleMemoryAdapter": PleMemoryAdapter,
            "TargetReaderHook": TargetReaderHook,
            "install_target_reader_hook": install_target_reader_hook,
            "install_bundle_adapter": install_bundle_adapter,
            "install_vllm_target_reader": install_vllm_target_reader,
            "install_sglang_target_reader": install_sglang_target_reader,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __repr__() -> str:
    backend = "pyo3" if _USING_PYO3 else ("ctypes" if _USING_CTYPES else "unavailable")
    return f"<engramdb {__version__} {backend} bindings>"
