#!/usr/bin/env python3
"""Real-table Arrow IPC smoke/validation.

This script is designed for machines that have the EngramDB real Store-I tree
present (e.g. /Volumes/My Passport/qwen38-rows or data/real-rows).  It:
  1. opens the real 128-shard Store,
  2. fetches a modest rowid sample,
  3. builds an Arrow table and IPC stream,
  4. reads the IPC stream back and verifies byte-for-byte content.

It is harmless to run on machines without the real table; it simply reports a
skip instead of failing (CI does not have the real table).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import engramdb

REAL_ROWS = Path(os.environ.get("ENGRAMDB_REAL_ROWS", "data/real-rows"))
SAMPLE = 4096


def main() -> int:
    if not (REAL_ROWS / "shard_000.bin").exists():
        print(f"[real-arrow] skip: {REAL_ROWS} not present")
        return 0

    try:
        import pyarrow as pa
    except ImportError:
        print("[real-arrow] skip: pyarrow not installed")
        return 0

    from engramdb.arrow_utils import store_fetch_arrow, table_to_ipc_bytes

    store = engramdb.Store(str(REAL_ROWS), 128, 2_500_012, 160)
    try:
        rowids = list(range(SAMPLE))
        table = store_fetch_arrow(store, rowids)
        assert table.num_rows == SAMPLE, table.num_rows
        assert table.column_names == ["rowid", "row"]
        raw = store.fetch(rowids)
        # Verify the row column contains exactly the same bytes as Store.fetch.
        arrow_bytes = b"".join(table.column("row").to_pylist())
        assert arrow_bytes == raw, (len(arrow_bytes), len(raw))
        ipc = table_to_ipc_bytes(table)
        with pa.ipc.open_stream(ipc) as reader:
            restored = reader.read_all()
        assert restored.num_rows == SAMPLE
        assert b"".join(restored.column("row").to_pylist()) == raw
        print(
            f"[real-arrow] OK rows={SAMPLE} ipc_bytes={len(ipc)} "
            f"store_bytes={len(raw)}"
        )
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
