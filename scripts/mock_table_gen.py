#!/usr/bin/env python3
"""生成结构等价的合成 PLE 表（FP8, F8_E4M3），用于磁盘受限时的开发。

默认按真实分片结构生成（scale>1 时按比例缩小行数但保留分片组织）：
  - 128 个 shard（或 /scale 个），每个 [rows_per_shard, 160]
  - 附带 manifest/spec.json（数据结构与真实 extract_ple_spec 一致）
  - 每 shard 一个 .bin + .json，写入 data/mock-qwen38-ple/

用法:
    python3 scripts/mock_table_gen.py [--scale 16] [--out data/mock-qwen38-ple] [--seed 42]
"""
import argparse
import json
import numpy as np
from pathlib import Path

REAL = {
    "shards": 128,
    "rows_per_shard": 2_500_012,
    "width": 160,
    "dtype": "F8_E4M3",
    "multipliers": [23703573157769, 20109073645365, 8052911324071],
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=int, default=16,
                    help="行数与分片数整体缩小倍数（默认 16 -> 约 3.2 亿/16 行）")
    ap.add_argument("--out", type=Path, default=Path("data/mock-qwen38-ple"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    n_shards = max(1, REAL["shards"] // args.scale)
    rows = (REAL["rows_per_shard"] + args.scale - 1) // args.scale
    width = REAL["width"]
    rng = np.random.default_rng(args.seed)

    args.out.mkdir(parents=True, exist_ok=True)
    spec = {
        "model": "mock-qwen38-ple",
        "scale": args.scale,
        "ngram_size": 3,
        "ngram_vocab_size_base": 20_000_000,
        "heads_per_ngram": 8,
        "ple_embed_dim": 2560,
        "ple_layer_ids": [2],
        "split_ngram_parts": n_shards,
        "make_ngram_vocab_size_divisible_by": 128,
        "ple_conv_kernel_size": 4,
        "hidden_size": 2560,
        "hc_count": 4,
        "vocab_size": 248_320,
        "layer_multipliers_i64": REAL["multipliers"],
        "shard_rows": rows,
        "shard_width": width,
        "shard_dtype": REAL["dtype"],
        "total_rows": rows * n_shards,
        "nsides": {"n_shards": n_shards},
    }
    (args.out / "spec.json").write_text(json.dumps(spec, indent=2, ensure_ascii=False))

    total = 0
    for i in range(n_shards):
        data = rng.integers(0, 256, size=(rows, width), dtype=np.uint8)
        path = args.out / f"shard_{i:03d}.bin"
        data.tofile(path)
        total += rows
    # 记录布局清单（与抽取脚本输出一致）
    layout = {
        "type": "batch-sharded",
        "shard_pattern": "shard_{:03d}.bin",
        "rows_per_shard": rows,
        "width": width,
        "n_shards": n_shards,
    }
    (args.out / "layout.json").write_text(json.dumps(layout, indent=2))
    print(f"mock table written: {args.out}  ({n_shards} shards x {rows} x {width} f8, "
          f"{total*width} elements = {total*width/1e9:.2f}G)")


if __name__ == "__main__":
    raise SystemExit(main())
