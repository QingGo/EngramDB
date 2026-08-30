"""Optional Arrow helpers for EngramDB batch reads.

These helpers are intentionally optional: they only require pyarrow when called.
They give products a zero-copy-friendly way to hand EngramDB rows to Arrow
consumers (pandas, DuckDB, Arrow IPC, etc.).
"""

from __future__ import annotations

from typing import Any, Iterable

from . import Store, View


def _pyarrow() -> Any:
    try:
        import pyarrow as pa
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "pyarrow is required for Arrow helpers; install it (e.g. pip install pyarrow)"
        ) from exc
    return pa


def store_fetch_arrow(store: Store, rowids: Iterable[int]) -> Any:
    """Return an Arrow table with one row per input rowid.

    Columns:
      - rowid: int64
      - row:   fixed-size binary (store.width bytes)
    """
    pa = _pyarrow()
    rowids_list = list(rowids)
    raw = store.fetch(rowids_list)
    width = store.width
    rows = [
        raw[i * width:(i + 1) * width]
        for i in range(len(rowids_list))
    ]
    return pa.table({
        "rowid": pa.array(rowids_list, type=pa.int64()),
        "row": pa.array(rows, type=pa.binary(width)),
    })


def view_read_arrow(view: View, indices: Iterable[int]) -> Any:
    """Return an Arrow table containing view records.

    Columns:
      - index: int64
      - slot:  fixed-size binary (view.slot_bytes)
    """
    pa = _pyarrow()
    indices_list = list(indices)
    slots = [view.read_record(i) for i in indices_list]
    return pa.table({
        "index": pa.array(indices_list, type=pa.int64()),
        "slot": pa.array(slots, type=pa.binary(view.slot_bytes)),
    })


def table_to_ipc_bytes(table: Any) -> bytes:
    """Serialize an Arrow table to an Arrow IPC stream (bytes)."""
    pa = _pyarrow()
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue().to_pybytes()
