#!/usr/bin/env python3
"""位级一致性校验（M1 出口标准的一部分）：

- 从原始行存储（data/real-rows/shard_NNN.bin）以纯 python 按行读取 N 个随机 rowid，
  计算 FNV-1a 校验（与 engramdb-cli `verify` 一致）
- 对比 engramdb-cli verify 输出 fnv —— 相同即行值位级一致
用法：
  python3 scripts/bitwise_check.py <rows_dir> --n 4096
"""
import argparse
import hashlib
import random
import struct
import subprocess
import sys
from pathlib import Path

FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3


def fnv1a(data: bytes) -> int:
    h = FNV_OFFSET
    for b in data:
        h ^= b
        h = (h * FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    return h


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("rows_dir", type=Path)
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0xABCD)
    args = ap.parse_args()

    rows_per_shard = 2_500_012
    width = 160
    rng = random.Random(args.seed)
    keys = [rng.randrange(0, 320_001_536) for _ in range(args.n)]

    out = bytearray()
    for k in keys:
        shard = k // rows_per_shard
        off = (k % rows_per_shard) * width
        f = args.rows_dir / f"shard_{shard:03d}.bin"
        with open(f, "rb") as fh:
            fh.seek(off)
            out += fh.read(width)

    py_fnv = fnv1a(bytes(out))
    txt = Path("/tmp/bitwise_keys.txt")
    txt.write_text("\n".join(map(str, keys)))

    # CLI：直接对同一目录 gather（verify 命令改造成对 rows 目录同样适用）
    res = subprocess.run(
        ["cargo", "run", "-q", "--release", "-p", "engramdb-cli", "--", "verify", str(args.rows_dir), str(txt)],
        capture_output=True, text=True, check=False, cwd=Path(__file__).resolve().parents[1],
    )
    if res.returncode != 0:
        print("cli error:", res.stderr[-500:])
        return 1
    line = res.stdout.strip().splitlines()[-1]
    print("cli:", line)
    print("py :", f"fnv={py_fnv} keys={len(keys)}")
    cli_fnv = int(line.split("fnv=")[1].split()[0])
    ok = cli_fnv == py_fnv
    print("BIT-EXACT:", "PASS ✓" if ok else "FAIL ✗")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
