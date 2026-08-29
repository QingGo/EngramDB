#!/usr/bin/env python3
"""从真实 FP8 分片中提取 ngram_embedding 张量 → 原始行文件 shard_NNN.bin（每行 160B FP8）。

输出：<out>/shard_000..127.bin（与 engramdb-core Layout(128, 2500012,160,1) 直接对应）
全程顺序读 + 顺序写；预计 ~10min（50GB）。
"""
import json
import struct
import sys
from pathlib import Path

HEADERS = {}


def load_header(path: Path):
    global HEADERS
    if path not in HEADERS:
        with open(path, "rb") as f:
            L = struct.unpack("<Q", f.read(8))[0]
            h = json.loads(f.read(L))
            HEADERS[path] = (h, L + 8)  # data start = 8(json len) + json bytes
    return HEADERS[path]


def main() -> int:
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "data/qwen38-ple-fp8")
    dst = Path(sys.argv[2] if len(sys.argv) > 2 else "data/real-rows")
    dst.mkdir(parents=True, exist_ok=True)
    idx = json.loads((src / "model.safetensors.index.json").read_text())
    wm = idx["weight_map"]
    shard_files = {
        i: wm[f"model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_{i}.weight"]
        for i in range(128)
    }

    total = 0
    for i, fname in shard_files.items():
        fpath = src / fname
        h, data_start = load_header(fpath)
        tname = f"model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_{i}.weight"
        entry = h[tname]
        start, end = entry["data_offsets"]
        rows, width = entry["shape"]
        assert (rows, width) == (2_500_012, 160), (i, rows, width)
        nbytes = end - start
        out = dst / f"shard_{i:03d}.bin"
        if out.exists() and out.stat().st_size == nbytes:
            continue  # resume
        with open(fpath, "rb") as f:
            f.seek(data_start + start)
            with open(out, "wb") as g:
                remaining = nbytes
                while remaining > 0:
                    buf = f.read(min(64 << 20, remaining))
                    if not buf:
                        break
                    g.write(buf)
                    remaining -= len(buf)
        total += nbytes
        if i % 16 == 0 or i == 127:
            print(f"{i:3d}/128  wrote {out.name} ({nbytes/1e9:.3f} GB)", flush=True)
    print(f"done: {total/1e9:.2f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
