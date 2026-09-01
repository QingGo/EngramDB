"""Disk-backed, scalable rowid-tuple -> Store-P slot index.

This is the Phase B answer to V133.  The in-memory :class:`SlotIndex` keeps the
full rowid matrix and sorted arrays in RAM, which is fine for 1M-10M grams but
not for a full 320M-gram Store-P table.  :class:`DiskSlotIndex` uses a bucketed
on-disk index:

* records are (`16 x u64` rowid tuple, `u64` slot), fixed 136 bytes;
* rowid tuples are hashed to one of N buckets;
* the build streams the input twice: first to count bucket sizes, then to place
  each record into its bucket region;
* each bucket is sorted independently and written as a small file;
* lookup hashes to a bucket, keeps a bounded LRU of loaded buckets, and binary
  searches inside that bucket.

Resident memory is bounded by ``cache_buckets * average_bucket_size`` while
keeping exact lookups and supporting 320M-scale tables.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
from collections import OrderedDict
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

_FORMAT = "engramdb-disk-slot-index-v1"
_RECORD_BYTES = 16 * 8 + 8


def _row_key(row: np.ndarray | tuple[int, ...] | list[int]) -> bytes:
    return struct.pack("<16Q", *(int(x) for x in row))


def _bucket_id(key: bytes, num_buckets: int) -> int:
    digest = hashlib.blake2b(key, digest_size=8).digest()
    return int.from_bytes(digest, "little") % num_buckets


class DiskSlotIndex:
    """Exact disk-backed rowid -> slot index with a bounded bucket LRU."""

    def __init__(self, directory: str | Path, *, cache_buckets: int = 64) -> None:
        self.directory = Path(directory)
        meta_path = self.directory / "index.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"disk slot index not found: {meta_path}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("format") != _FORMAT:
            raise ValueError(f"unsupported disk slot index format: {meta.get('format')}")
        self.heads = int(meta["heads"])
        self.num_buckets = int(meta["num_buckets"])
        self.count = int(meta["count"])
        self.cache_buckets = max(1, int(cache_buckets))
        self._record_bytes = _RECORD_BYTES
        self._cache: OrderedDict[int, list[bytes]] = OrderedDict()
        self._closed = False

    @classmethod
    def build(
        cls,
        rowids: Iterable[tuple[int, ...] | np.ndarray] | np.ndarray,
        output_dir: str | Path,
        *,
        heads: int = 16,
        num_buckets: int = 16384,
        cache_buckets: int = 64,
        slots: Iterable[int] | None = None,
    ) -> DiskSlotIndex:
        """Build a disk index from an iterable of rowid tuples.

        ``rowids`` may be a ``[N, heads]`` numpy array or any iterable yielding
        length-``heads`` tuples.  ``slots`` defaults to ``arange(N)``.
        """
        if heads != 16:
            raise NotImplementedError("DiskSlotIndex currently supports heads=16")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        buckets_dir = output_dir / "buckets"
        buckets_dir.mkdir(exist_ok=True)
        temp_dir = output_dir / ".tmp"
        temp_dir.mkdir(exist_ok=True)

        raw_path = temp_dir / "raw.bin"
        grouped_path = temp_dir / "grouped.bin"
        rec_bytes = _RECORD_BYTES
        counts = np.zeros(num_buckets, dtype=np.int64)
        count = 0
        slot_iter = iter(slots) if slots is not None else None

        # Pass 1: stream every record to a raw file, count bucket occupancy.
        with open(raw_path, "wb") as raw:
            for i, row in enumerate(rowids):
                key = _row_key(row)
                slot = i if slot_iter is None else int(next(slot_iter))
                bucket = _bucket_id(key, num_buckets)
                raw.write(key + struct.pack("<Q", slot))
                counts[bucket] += 1
                count += 1

        # Compute byte offsets of each bucket region in the grouped file.
        offsets = np.zeros(num_buckets + 1, dtype=np.int64)
        np.cumsum(counts * rec_bytes, out=offsets[1:])

        # Pass 2: place each record into its bucket region in the grouped file.
        grouped_size = int(offsets[-1])
        fd = os.open(grouped_path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
        try:
            os.ftruncate(fd, grouped_size)
            cursor = np.zeros(num_buckets, dtype=np.int64)
            with open(raw_path, "rb") as raw:
                while True:
                    record = raw.read(rec_bytes)
                    if not record:
                        break
                    key = record[:128]
                    bucket = _bucket_id(key, num_buckets)
                    pos = int(offsets[bucket] + cursor[bucket] * rec_bytes)
                    os.pwrite(fd, record, pos)
                    cursor[bucket] += 1

            # Sort each bucket and write its small file.
            with open(grouped_path, "rb") as grouped:
                for bucket in range(num_buckets):
                    start = int(offsets[bucket])
                    end = int(offsets[bucket + 1])
                    if start == end:
                        continue
                    grouped.seek(start)
                    data = grouped.read(end - start)
                    records = [
                        data[i : i + rec_bytes] for i in range(0, len(data), rec_bytes)
                    ]
                    records.sort()
                    (buckets_dir / f"{bucket:04d}.bin").write_bytes(b"".join(records))
        finally:
            os.close(fd)

        meta = {
            "format": _FORMAT,
            "heads": heads,
            "num_buckets": num_buckets,
            "count": count,
            "record_bytes": rec_bytes,
            "cache_buckets": cache_buckets,
        }
        (output_dir / "index.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        # Remove temporary files.
        for p in temp_dir.iterdir():
            try:
                p.unlink()
            except OSError:
                pass
        try:
            temp_dir.rmdir()
        except OSError:
            pass

        return cls(output_dir, cache_buckets=cache_buckets)

    @classmethod
    def open(cls, directory: str | Path, *, cache_buckets: int = 64) -> DiskSlotIndex:
        return cls(directory, cache_buckets=cache_buckets)

    def __len__(self) -> int:
        return self.count

    def _load_bucket(self, bucket: int) -> list[bytes]:
        if bucket in self._cache:
            self._cache.move_to_end(bucket)
            return self._cache[bucket]
        path = self.directory / "buckets" / f"{bucket:04d}.bin"
        if not path.exists():
            return []
        data = path.read_bytes()
        rec_bytes = self._record_bytes
        records = [
            data[i : i + rec_bytes] for i in range(0, len(data), rec_bytes)
        ]
        self._cache[bucket] = records
        if len(self._cache) > self.cache_buckets:
            self._cache.popitem(last=False)
        return records

    def lookup(self, row: tuple[int, ...] | np.ndarray) -> int:
        if self._closed:
            raise ValueError("DiskSlotIndex is closed")
        key = _row_key(row)
        bucket = _bucket_id(key, self.num_buckets)
        records = self._load_bucket(bucket)
        import bisect

        idx = bisect.bisect_left(records, key)
        if idx < len(records) and records[idx].startswith(key):
            return int.from_bytes(records[idx][128:], "little")
        raise KeyError(f"rowid tuple not found in disk slot index: {tuple(int(x) for x in row)}")

    def lookup_all(self, row: tuple[int, ...] | np.ndarray) -> list[int]:
        if self._closed:
            raise ValueError("DiskSlotIndex is closed")
        key = _row_key(row)
        bucket = _bucket_id(key, self.num_buckets)
        records = self._load_bucket(bucket)
        import bisect

        out: list[int] = []
        idx = bisect.bisect_left(records, key)
        while idx < len(records) and records[idx].startswith(key):
            out.append(int.from_bytes(records[idx][128:], "little"))
            idx += 1
        if not out:
            raise KeyError(f"rowid tuple not found in disk slot index: {tuple(int(x) for x in row)}")
        return out

    def to_slots(self, rowids: np.ndarray) -> np.ndarray:
        rows = np.asarray(rowids, dtype=np.int64)
        if rows.ndim != 2:
            raise ValueError(f"rowids must be [N, {self.heads}], got {rows.shape}")
        out = np.empty(len(rows), dtype=np.int64)
        for i, row in enumerate(rows):
            out[i] = self.lookup(tuple(int(x) for x in row))
        return out

    def memory_bytes(self) -> int:
        return sum(len(records) * self._record_bytes for records in self._cache.values())

    def stats(self) -> dict[str, int]:
        return {
            "count": self.count,
            "num_buckets": self.num_buckets,
            "cached_buckets": len(self._cache),
            "cached_bytes": self.memory_bytes(),
        }

    def close(self) -> None:
        self._cache.clear()
        self._closed = True

    def __enter__(self) -> DiskSlotIndex:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    @classmethod
    def build_from_keys_file(
        cls,
        keys_path: str | Path,
        output_dir: str | Path,
        *,
        heads: int = 16,
        num_buckets: int = 16384,
        cache_buckets: int = 64,
    ) -> DiskSlotIndex:
        """Build from an EngramDB flat keys file without loading it into RAM."""

        def rows() -> Iterator[tuple[int, ...]]:
            buf: list[int] = []
            with open(keys_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    buf.append(int(line))
                    if len(buf) == heads:
                        yield tuple(buf)
                        buf = []
            if buf:
                raise ValueError(
                    f"keys file has incomplete rowid tuple: {len(buf)} values"
                )

        return cls.build(
            rows(),
            output_dir,
            heads=heads,
            num_buckets=num_buckets,
            cache_buckets=cache_buckets,
        )
