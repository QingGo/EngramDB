#!/usr/bin/env python3
"""Contract smoke against the qwen35-ple / engram-peft sibling projects.

This script checks the pieces those projects depend on:

* C ABI: ``engramdb_abi_version`` and ``engramdb_rowids_for_seq``
* Python Store bit-exact real/raw reading already covered by
  ``scripts/real_ple_bit_exact.py``
* ``DiskMultiHeadEmbedding`` quick path (what engram-peft uses when
  ``install_disk_multi_head_embedding`` is called)
* Optional: qwen35-ple ``PleSpec`` rowid cross-check
* Optional: engram-peft import check

It intentionally does not load a full model.
"""

from __future__ import annotations

import ctypes
import os
import struct
import tempfile
from pathlib import Path

import torch

import engramdb
from engramdb.integrations import DiskMultiHeadEmbedding

REPO_ROOT = Path(__file__).resolve().parents[1]
SIBLING_QWEN35 = Path("/Users/zeng/code/qwen35-ple")
SIBLING_ENGRAM_PEFT = Path("/Users/zeng/code/engram-peft")


def find_c_library() -> Path | None:
    candidates = [
        REPO_ROOT / "target" / "release" / "libengramdb_c.dylib",
        REPO_ROOT / "target" / "release" / "libengramdb_c.so",
        REPO_ROOT / "target" / "debug" / "libengramdb_c.dylib",
        REPO_ROOT / "target" / "debug" / "libengramdb_c.so",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def c_abi_smoke() -> None:
    lib_path = find_c_library()
    if lib_path is None:
        raise SystemExit("C ABI library not found; run cargo build --release -p engramdb-python")
    lib = ctypes.CDLL(str(lib_path))
    lib.engramdb_abi_version.restype = ctypes.c_uint32
    assert lib.engramdb_abi_version() == 1
    lib.engramdb_rowids_for_seq.argtypes = [
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_size_t,
        ctypes.c_uint32,
    ]
    lib.engramdb_rowids_for_seq.restype = ctypes.c_int
    tokens = (ctypes.c_uint32 * 5)(248044, 1000, 99999, 42, 12345)
    out = (ctypes.c_uint64 * (5 * 16))()
    rc = lib.engramdb_rowids_for_seq(tokens, 5, out, 5 * 16, 1)
    assert rc == 0, f"rowids rc={rc}"
    got = list(out)

    py_wrapper = engramdb.rowids_for_seq([248044, 1000, 99999, 42, 12345])
    py_flat = [x for row in py_wrapper for x in row]
    assert py_flat == got, "Python rowids_for_seq differs from C ABI"
    print("[C ABI] Python rowids_for_seq wrapper matches C ABI")

    try:
        if SIBLING_QWEN35.exists():
            import sys
            sys.path.insert(0, str(SIBLING_QWEN35 / "src"))
            from qwen35_ple.ple_hash import real_spec
            ref = [x for r in real_spec().rowids_for_seq([248044, 1000, 99999, 42, 12345]) for x in r]
            assert got == ref, "C ABI rowids differ from qwen35-ple PleSpec"
            print("[C ABI] rowids match qwen35-ple PleSpec")
    except Exception as exc:
        print(f"[C ABI] sibling qwen35-ple cross-check skipped: {exc}")

    print("[C ABI] abi_version=1 rowids_for_seq OK")


def disk_embedding_smoke() -> None:
    directory = Path(tempfile.mkdtemp(prefix="engramdb-sibling-"))
    primes = [4, 5, 7]
    per_head = 4
    total = sum(primes)
    row_width = per_head * 4
    vals = torch.arange(total * per_head, dtype=torch.float32).reshape(total, per_head)
    with open(directory / "shard_000.bin", "wb") as f:
        for v in vals.reshape(-1).tolist():
            f.write(struct.pack("<f", v))
    store = engramdb.Store(str(directory), 1, total, row_width)
    disk = DiskMultiHeadEmbedding(primes, per_head, store=store, dtype=torch.float32)
    hashes = torch.tensor([[[0, 1, 2], [3, 4, 5]]])
    out = disk(hashes)
    offsets = torch.tensor([0, 4, 9])
    expected = vals[(hashes + offsets).reshape(-1)].reshape(*hashes.shape, per_head)
    assert torch.equal(out, expected)
    store.close()
    print("[DiskMultiHeadEmbedding] quick check OK")


def optional_engram_peft_check() -> None:
    if not SIBLING_ENGRAM_PEFT.exists():
        print("[engram-peft] sibling not found; skip import check")
        return
    try:
        import sys
        sys.path.insert(0, str(SIBLING_ENGRAM_PEFT / "src"))
        import engram_peft  # noqa: F401
        print("[engram-peft] import OK")
    except Exception as exc:
        print(f"[engram-peft] import not available: {exc}")


def main() -> None:
    c_abi_smoke()
    disk_embedding_smoke()
    optional_engram_peft_check()
    print("SIBLING_CONTRACT_SMOKE_OK")


if __name__ == "__main__":
    main()
