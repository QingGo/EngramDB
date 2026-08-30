
"""Minimal vLLM-oriented disk gather helper.

This module is intentionally engine-agnostic: it provides the same primitives
that a vLLM PLE mmap patch would need (dedup, batched disk fetch, expansion),
so a future vLLM plugin can use EngramDB instead of raw ``np.memmap`` gathers.
"""

from __future__ import annotations

from typing import Iterable

from . import Store


class PleDiskGather:
    """Dedup + EngramDB batch fetch + expand to original row order."""

    def __init__(self, store: Store, row_bytes: int):
        self.store = store
        self.row_bytes = row_bytes

    def fetch(self, rowids: Iterable[int]) -> bytes:
        rowids_list = list(rowids)
        if not rowids_list:
            return b""

        # Dedup while preserving order.  This mirrors the vLLM mmap patch's
        # ``np.unique`` optimization: only unique rows hit the disk.
        seen: set[int] = set()
        unique: list[int] = []
        for r in rowids_list:
            if r not in seen:
                seen.add(r)
                unique.append(r)

        raw = self.store.fetch(unique)
        if len(raw) != len(unique) * self.row_bytes:
            raise RuntimeError(
                f"EngramDB fetch returned {len(raw)} bytes for "
                f"{len(unique)} rows x {self.row_bytes}"
            )

        # Expand back into the original access order.
        index = {r: i for i, r in enumerate(unique)}
        return b"".join(
            raw[index[r] * self.row_bytes:index[r] * self.row_bytes + self.row_bytes]
            for r in rowids_list
        )

    def fetch_unique(self, rowids: Iterable[int]) -> bytes:
        """Return only unique rows in first-seen order (for staging buffers)."""
        seen: set[int] = set()
        unique: list[int] = []
        for r in rowids:
            if r not in seen:
                seen.add(r)
                unique.append(r)
        return self.store.fetch(unique)
