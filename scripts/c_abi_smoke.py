#!/usr/bin/env python3
"""Pure-stdlib C ABI smoke for EngramDB.

This is the CI-friendly version of ``sibling_contract_smoke.py``: it does not
require PyTorch or any sibling repository, only the compiled
``libengramdb_c`` cdylib.

It checks:

* ``engramdb_abi_version() == 1``
* ``engramdb_rowids_for_seq`` matches the checked-in Rust golden rowids
"""

from __future__ import annotations

import ctypes
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


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


def main() -> int:
    lib_path = find_c_library()
    if lib_path is None:
        print("C ABI library not found; run: cargo build --release -p engramdb-python")
        return 1

    lib = ctypes.CDLL(str(lib_path))
    lib.engramdb_abi_version.restype = ctypes.c_uint32
    assert lib.engramdb_abi_version() == 1, "unexpected ABI version"

    lib.engramdb_rowids_for_seq.argtypes = [
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_size_t,
        ctypes.c_uint32,
    ]
    lib.engramdb_rowids_for_seq.restype = ctypes.c_int

    golden = json.loads(
        (REPO_ROOT / "crates" / "engramdb-keygen" / "tests" / "golden.json").read_text()
    )
    tokens = [int(x) for x in golden["tokens"]]
    expected = [[int(x) for x in row] for row in golden["rowids"]]

    arr = (ctypes.c_uint32 * len(tokens))(*tokens)
    out = (ctypes.c_uint64 * (len(tokens) * 16))()
    rc = lib.engramdb_rowids_for_seq(arr, len(tokens), out, len(tokens) * 16, 1)
    assert rc == 0, f"engramdb_rowids_for_seq rc={rc}"

    got = [list(out[i * 16:(i + 1) * 16]) for i in range(len(tokens))]
    assert got == expected, "C ABI rowids differ from Rust golden"

    print(f"[C ABI] abi_version=1, {len(tokens)} rowids match golden")
    print("C_ABI_SMOKE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
