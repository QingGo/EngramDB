"""Multi-table registry for EngramDB.

A table is simply a subdirectory containing EngramDB shard files.  This module
provides the small amount of bookkeeping needed to serve several tables from one
root directory without changing the storage format.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from . import Store


class Database:
    """Open multiple EngramDB stores from one root directory."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self._stores: dict[tuple[str, int, int, int], Store] = {}
        self._pools: dict[tuple[str, int, int, int], Any] = {}

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

    def open_pool(
        self,
        table: str,
        shards: int,
        rows_per_shard: int,
        width: int,
        pool_size: int = 4,
    ) -> Any:
        """Open (and cache) a thread-safe StorePool for a named table."""
        from .pool import StorePool

        key = (table, shards, rows_per_shard, width)
        if key not in self._pools:
            path = self.root / table
            self._pools[key] = StorePool(
                str(path),
                shards,
                rows_per_shard,
                width,
                pool_size=pool_size,
            )
        return self._pools[key]

    def fetch(
        self,
        table: str,
        rowids: Iterable[int],
        shards: int,
        rows_per_shard: int,
        width: int,
    ) -> bytes:
        # Use a pooled thread-safe handle when available; each request borrows
        # one Store from the pool instead of opening/closing every time.
        pool = self.open_pool(table, shards, rows_per_shard, width)
        with pool as store:
            return store.fetch(list(rowids))

    def close(self) -> None:
        for store in self._stores.values():
            store.close()
        self._stores.clear()
        for pool in self._pools.values():
            pool.close()
        self._pools.clear()
