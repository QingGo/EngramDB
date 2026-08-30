"""Multi-table registry for EngramDB.

A table is simply a subdirectory containing EngramDB shard files.  This module
provides the small amount of bookkeeping needed to serve several tables from one
root directory without changing the storage format.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from . import Store


class Database:
    """Open multiple EngramDB stores from one root directory."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self._stores: dict[tuple[str, int, int, int], Store] = {}

    def list_tables(self) -> list[str]:
        """Return table ids that contain an EngramDB shard file."""
        if not self.root.exists():
            return []
        out: list[str] = []
        for p in sorted(self.root.iterdir()):
            if p.is_dir() and any(p.glob("shard_*.bin")):
                out.append(p.name)
        return out

    def open_store(
        self,
        table: str,
        shards: int,
        rows_per_shard: int,
        width: int,
    ) -> Store:
        """Open (and cache) a `Store` for a named table."""
        key = (table, shards, rows_per_shard, width)
        if key not in self._stores:
            path = self.root / table
            self._stores[key] = Store(str(path), shards, rows_per_shard, width)
        return self._stores[key]

    def fetch(
        self,
        table: str,
        rowids: Iterable[int],
        shards: int,
        rows_per_shard: int,
        width: int,
    ) -> bytes:
        # Open a fresh Store in the calling thread.  The native Store is
        # unsendable, so sharing cached stores across server threads is unsafe.
        store = Store(str(self.root / table), shards, rows_per_shard, width)
        try:
            return store.fetch(list(rowids))
        finally:
            store.close()

    def close(self) -> None:
        for store in self._stores.values():
            store.close()
        self._stores.clear()
