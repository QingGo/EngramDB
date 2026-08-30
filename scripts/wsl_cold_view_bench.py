#!/usr/bin/env python3
"""Cold-cache view read benchmark: sequential vs random on WSL/Linux.

Requires a Store-P view produced by:
    engramdb view build ... /tmp/access.view /tmp/access.keys --keys ...
Uses posix_fadvise(DONTNEED) before each run to approximate cold reads.
"""
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

VIEW = sys.argv[1] if len(sys.argv) > 1 else "/tmp/access.view"
SLOT = int(sys.argv[2]) if len(sys.argv) > 2 else 2560
GRAMS = int(sys.argv[3]) if len(sys.argv) > 3 else 19999
THREADS = int(sys.argv[4]) if len(sys.argv) > 4 else 8

fd = os.open(VIEW, os.O_RDONLY)
file_size = os.fstat(fd).st_size
if file_size != GRAMS * SLOT:
    print(f"warning: file size {file_size} != grams*slot {GRAMS*SLOT}")


def random_order(n: int, seed: int = 0xCAFE_BEEF) -> list[int]:
    state = seed
    out = []
    for _ in range(n):
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        out.append(state % n)
    return out


def evict() -> None:
    # Drop clean page cache for this file.
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)


def read_range(indices: list[int], threads: int) -> tuple[float, int]:
    chunk = (len(indices) + threads - 1) // threads

    def worker(lo: int, hi: int) -> int:
        n = 0
        for idx in indices[lo:hi]:
            data = os.pread(fd, SLOT, idx * SLOT)
            n += len(data)
        return n

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=threads) as ex:
        futures = [
            ex.submit(worker, i * chunk, min((i + 1) * chunk, len(indices)))
            for i in range(threads)
        ]
        total = sum(f.result() for f in futures)
    dt = time.perf_counter() - t0
    return dt, total


def run(name: str, indices: list[int], threads: int) -> None:
    evict()
    time.sleep(0.05)
    dt, total = read_range(indices, threads)
    rows = len(indices)
    rps = rows / dt
    mbps = total / 1e6 / dt
    print(f"{name}: rows={rows} time={dt:.3f}s rows/s={rps:.0f} MB/s={mbps:.1f}")


seq = list(range(GRAMS))
rand = random_order(GRAMS)

print(f"view={VIEW} slot={SLOT} grams={GRAMS} threads={THREADS}")
run("COLD SEQ 1t", seq, 1)
run("COLD RAND 1t", rand, 1)
run("COLD SEQ 8t", seq, THREADS)
run("COLD RAND 8t", rand, THREADS)

os.close(fd)
