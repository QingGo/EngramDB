"""Thread-safe EngramDB Store handle pool.

EngramDB's native Store is safe for concurrent use once opened, but long-lived
services and multi-worker training benefit from explicit handle lifecycle
management.  This module provides a small bounded pool that can be used as a
context manager:

    with StorePool(dir, shards, rows_per_shard, width, size=4) as store:
        data = store.fetch(rowids)

or with explicit acquire/release for thread-local borrowing:

    handle = pool.acquire()
    try:
        ...
    finally:
        pool.release(handle)
"""

from __future__ import annotations

import queue
import threading
from typing import Any, Iterator

from . import Store


class StorePool:
    """A bounded pool of open :class:`engramdb.Store` handles."""

    def __init__(
        self,
        directory: str,
        shards: int,
        rows_per_shard: int,
        width: int,
        *,
        pool_size: int = 4,
    ) -> None:
        self.directory = str(directory)
        self.shards = int(shards)
        self.rows_per_shard = int(rows_per_shard)
        self.width = int(width)
        self.pool_size = max(1, int(pool_size))
        self._queue: queue.LifoQueue[Store] = queue.LifoQueue(
            maxsize=self.pool_size
        )
        self._all: list[Store] = []
        self._lock = threading.Lock()
        self._closed = False
        for _ in range(self.pool_size):
            self._queue.put_nowait(self._open())

    def _open(self) -> Store:
        store = Store(
            self.directory,
            shards=self.shards,
            rows_per_shard=self.rows_per_shard,
            width=self.width,
        )
        self._all.append(store)
        return store

    def acquire(self) -> Store:
        """Return an idle handle, blocking until one is available."""
        if self._closed:
            raise RuntimeError("StorePool is closed")
        return self._queue.get()

    def release(self, store: Store) -> None:
        """Return a handle to the pool, or close it if the pool is closed."""
        if self._closed:
            store.close()
            return
        self._queue.put(store)

    def __enter__(self) -> Store:
        self._current = self.acquire()
        return self._current

    def __exit__(self, *exc_info: Any) -> None:
        release = getattr(self, "_current", None)
        if release is not None:
            self.release(release)
            self._current = None

    @property
    def size(self) -> int:
        return self.pool_size

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with self._lock:
            while True:
                try:
                    store = self._queue.get_nowait()
                except queue.Empty:
                    break
                store.close()
            for store in self._all:
                try:
                    store.close()
                except Exception:
                    pass
            self._all.clear()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class ThreadLocalStore:
    """Convenience wrapper that keeps one pooled handle per thread."""

    def __init__(self, pool: StorePool) -> None:
        self.pool = pool
        self._local = threading.local()

    def get(self) -> Store:
        store = getattr(self._local, "store", None)
        if store is None:
            store = self.pool.acquire()
            self._local.store = store
        return store

    def release_current(self) -> None:
        store = getattr(self._local, "store", None)
        if store is not None:
            self.pool.release(store)
            self._local.store = None

    def close(self) -> None:
        self.release_current()
        self.pool.close()

    def __enter__(self) -> "ThreadLocalStore":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()


__all__ = ["StorePool", "ThreadLocalStore"]
