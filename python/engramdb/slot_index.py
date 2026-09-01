"""Generic rowid-tuple to Store-P slot semantic index.

A Store-P view is a flat array of fixed-size slots.  Each slot contains the
complete ``[16, 160]`` FP8 record for one PLE gram (one token position).  The
raw view format knows only physical slot numbers; it does not know which
rowid tuple (or which token) produced a slot.

:class:`SlotIndex` is the missing semantic bridge.  It is built from the keys
file that EngramDB writes when constructing a view (one flat rowid per line,
16 rowids per gram, in slot order).  Given any rowid tuple, it returns the
physical slot that contains the equivalent PLE record.

Because the same rowid tuple may occur at several slots (duplicate n-grams in
an access-order corpus), ``lookup`` returns a deterministic representative
slot.  All slots for identical rowid tuples contain identical bytes, so any
representative is semantically correct.  :meth:`lookup_all` returns every
matching slot when occurrence-level scheduling is needed.

The on-disk format is a small ``.npz``:

* ``rowids``  : ``[N, heads]`` uint64 matrix in slot order
* ``slots``   : ``[N]`` int64 physical slot ids (usually ``arange(N)``)
* ``heads``   : scalar

Lookup uses a sorted void-key binary search instead of a Python dict, so it
scales to millions of grams without keeping a hash table in memory.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

SlotRow = tuple[int, ...]


def _read_keys_file(path: str | Path) -> list[int]:
    values: list[int] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                values.append(int(line))
    return values


class SlotIndex:
    """Map PLE rowid tuples to Store-P physical slots."""

    def __init__(
        self,
        rowids: np.ndarray,
        slots: np.ndarray | None = None,
        *,
        heads: int | None = None,
    ) -> None:
        arr = np.asarray(rowids, dtype=np.int64)
        if arr.ndim != 2:
            raise ValueError(f"rowids must be a 2-D [N, heads] array, got {arr.shape}")
        if arr.shape[1] == 0:
            raise ValueError("rowids must have at least one head")
        if heads is not None and int(heads) != arr.shape[1]:
            raise ValueError(
                f"heads={heads} does not match rowids shape {arr.shape[1]}"
            )
        self.rowids = arr
        self.heads = int(arr.shape[1])
        if slots is None:
            slots = np.arange(len(arr), dtype=np.int64)
        self.slots = np.asarray(slots, dtype=np.int64)
        if len(self.slots) != len(self.rowids):
            raise ValueError(
                f"slots length {len(self.slots)} != rowids length {len(self.rowids)}"
            )
        self._sorted_keys: np.ndarray | None = None
        self._sorted_slots: np.ndarray | None = None
        self._rebuild_sorted()

    @classmethod
    def from_rowids(
        cls,
        rowids: np.ndarray,
        slots: np.ndarray | None = None,
        *,
        heads: int | None = None,
    ) -> SlotIndex:
        return cls(rowids, slots, heads=heads)

    @classmethod
    def from_keys_file(
        cls,
        path: str | Path,
        *,
        heads: int = 16,
        slots: np.ndarray | None = None,
    ) -> SlotIndex:
        """Build from an EngramDB flat keys file (16 rowids per gram)."""
        values = _read_keys_file(path)
        if len(values) % heads != 0:
            raise ValueError(
                f"keys file has {len(values)} rowids, not a multiple of heads {heads}"
            )
        arr = np.asarray(values, dtype=np.int64).reshape(-1, heads)
        return cls.from_rowids(arr, slots=slots, heads=heads)

    @classmethod
    def build(
        cls,
        rowids: np.ndarray,
        slots: np.ndarray | None = None,
        *,
        heads: int | None = None,
    ) -> SlotIndex:
        """Alias for :meth:`from_rowids`."""
        return cls.from_rowids(rowids, slots=slots, heads=heads)

    @classmethod
    def from_keys(
        cls,
        path: str | Path,
        *,
        heads: int = 16,
        slots: np.ndarray | None = None,
    ) -> SlotIndex:
        """Alias for :meth:`from_keys_file`."""
        return cls.from_keys_file(path, heads=heads, slots=slots)

    @classmethod
    def load(cls, path: str | Path) -> SlotIndex:
        """Load from an ``.npz`` produced by :meth:`save`."""
        path = Path(path)
        with np.load(path) as data:
            rowids = np.asarray(data["rowids"], dtype=np.int64)
            slots = np.asarray(data["slots"], dtype=np.int64)
            heads = int(np.asarray(data["heads"]).reshape(())) if "heads" in data else rowids.shape[1]
        return cls(rowids, slots, heads=heads)

    def save(self, path: str | Path) -> Path:
        """Save as a portable ``.npz`` plus a small JSON sidecar."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            rowids=self.rowids,
            slots=self.slots,
            heads=np.asarray([self.heads], dtype=np.int64),
        )
        meta = {
            "format": "engramdb-slot-index-v1",
            "heads": self.heads,
            "count": len(self),
            "path": str(path),
        }
        path.with_suffix(".json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return path

    def __len__(self) -> int:
        return len(self.rowids)

    def _rebuild_sorted(self) -> None:
        if len(self.rowids) == 0:
            self._sorted_keys = np.array([], dtype=f"V{self.heads * 8}")
            self._sorted_slots = np.array([], dtype=np.int64)
            return
        keys = np.ascontiguousarray(self.rowids).view(f"V{self.heads * 8}").ravel()
        order = np.argsort(keys, kind="stable")
        self._sorted_keys = keys[order]
        self._sorted_slots = self.slots[order]

    def _key_for(self, row: SlotRow | np.ndarray) -> np.void:
        arr = np.asarray(row, dtype=np.int64)
        if arr.ndim == 0 or arr.shape[0] != self.heads:
            raise ValueError(f"expected a rowid tuple of length {self.heads}, got {arr.shape}")
        return np.ascontiguousarray(arr.reshape(1, self.heads)).view(
            f"V{self.heads * 8}"
        ).ravel()[0]

    def lookup(self, row: SlotRow | np.ndarray) -> int:
        """Return one physical slot for a rowid tuple.

        Raises ``KeyError`` if the tuple is not present in the view.
        """
        key = self._key_for(row)
        pos = int(np.searchsorted(self._sorted_keys, key))
        if pos < len(self._sorted_keys) and self._sorted_keys[pos] == key:
            return int(self._sorted_slots[pos])
        raise KeyError(f"rowid tuple not found in slot index: {tuple(int(x) for x in row)}")

    def lookup_all(self, row: SlotRow | np.ndarray) -> list[int]:
        """Return every slot whose rowid tuple exactly matches ``row``."""
        key = self._key_for(row)
        start = int(np.searchsorted(self._sorted_keys, key, side="left"))
        end = int(np.searchsorted(self._sorted_keys, key, side="right"))
        if start == end:
            raise KeyError(
                f"rowid tuple not found in slot index: {tuple(int(x) for x in row)}"
            )
        return [int(x) for x in self._sorted_slots[start:end]]

    def to_slots(self, rowids: np.ndarray) -> np.ndarray:
        """Map a ``[M, heads]`` rowid matrix to one representative slot each."""
        rows = np.asarray(rowids, dtype=np.int64)
        if rows.ndim != 2 or rows.shape[1] != self.heads:
            raise ValueError(
                f"rowids must be [M, {self.heads}], got {rows.shape}"
            )
        out = np.empty(len(rows), dtype=np.int64)
        for i, row in enumerate(rows):
            out[i] = self.lookup(tuple(int(x) for x in row))
        return out

    def map_rowids(self, rowids: np.ndarray) -> np.ndarray:
        """Alias for :meth:`to_slots`."""
        return self.to_slots(rowids)

    def lookup_many(self, rowids: np.ndarray) -> np.ndarray:
        """Alias for :meth:`to_slots`."""
        return self.to_slots(rowids)

    def semantic_slot_indices(
        self,
        tokens: np.ndarray,
        rowids: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Map token positions to slots and also return the rowid matrix.

        This is the convenience entry point used by the live training path:
        given a token stream and its ``[T, 16]`` rowids, return an array such
        that ``slot_indices[i]`` is the physical Store-P slot for token ``i``.
        """
        rows = np.asarray(rowids, dtype=np.int64)
        if len(rows) != len(tokens):
            raise ValueError(
                f"rowids length {len(rows)} != tokens length {len(tokens)}"
            )
        return self.to_slots(rows), rows

    def __contains__(self, row: SlotRow | np.ndarray) -> bool:
        try:
            self.lookup(row)
            return True
        except KeyError:
            return False

    def as_dict(self) -> dict[str, Any]:
        return {
            "heads": self.heads,
            "count": len(self),
            "rowids_shape": list(self.rowids.shape),
            "slots_shape": list(self.slots.shape),
        }
