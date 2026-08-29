#!/usr/bin/env python3
"""从真实 FP8 checkpoint 分片提取 PLE 精确规格（只读，不修改权重）。

用法:
    python3 scripts/extract_ple_spec.py [--shards-dir PATH] [--out PATH]

输出 JSON 规格（默认 stdout），供 keygen/布局/mock 生成器使用。
"""
import argparse
import json
import struct
import sys
from pathlib import Path


def read_tensor(path: Path, name: str, dtype: str, shape):
    """从 safetensors 文件按头部顺序定位并读取张量数据（读 header 内 data_offsets 或顺序累加）。"""
    with open(path, "rb") as f:
        blob_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(blob_len))
        data_start = blob_len + 8
        pos = data_start
        order = [k for k in header if not k.startswith("__")]
        dtype_sizes = {"BF16": 2, "F8_E4M3": 1, "I64": 8, "I32": 4, "F32": 4}
        for k in order:
            sz = 1
            for s in header[k]["shape"]:
                sz *= s
            nbytes = dtype_sizes.get(header[k]["dtype"], 4) * sz
            if k == name:
                break
            pos += nbytes
        return header, pos


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards-dir", type=Path, default=Path("data/qwen38-ple-fp8"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    d = args.shards_dir
    cfg = json.loads((d / "config.json").read_text())["text_config"]
    idx = json.loads((d / "model.safetensors.index.json").read_text())
    wm = idx["weight_map"]

    ple_files = sorted({f for t, f in wm.items() if "ple." in t})
    ngram_files = sorted({f for t, f in wm.items() if "ngram_embedding" in t})

    # 读取含 multiplier 的分片（已知在 model-00005）
    with open(d / "model-00005-of-00131.safetensors", "rb") as f:
        L = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(L))
        data_start = L + 8
        pos = data_start
        dtype_sizes = {"BF16": 2, "F8_E4M3": 1, "I64": 8, "I32": 4, "F32": 4}
        multipliers = None
        shard_shape = None
        for k in [x for x in header if not x.startswith("__")]:
            sz = 1
            for s in header[k]["shape"]:
                sz *= s
            nbytes = dtype_sizes.get(header[k]["dtype"], 4) * sz
            if "layer_multipliers" in k:
                import numpy as np
                multipliers = np.fromfile(
                    d / "model-00005-of-00131.safetensors",
                    dtype=np.int64,
                    count=sz,
                    offset=pos,
                ).tolist()
            if "ngram_embedding.shard_0.weight" in k:
                shard_shape = header[k]["shape"]
                shard_dtype = header[k]["dtype"]
            pos += nbytes

    spec = {
        "model": "Qwen/Qwen3.8-Flash-Next-FP8",
        "arch": cfg.get("model_type"),
        "ngram_size": cfg.get("ngram_size"),
        "ngram_vocab_size_base": cfg.get("ngram_vocab_size_base"),
        "heads_per_ngram": cfg.get("heads_per_ngram"),
        "ple_embed_dim": cfg.get("ple_embed_dim"),
        "ple_layer_ids": cfg.get("ple_layer_ids"),
        "split_ngram_parts": cfg.get("split_ngram_parts"),
        "make_ngram_vocab_size_divisible_by": cfg.get("make_ngram_vocab_size_divisible_by"),
        "ple_conv_kernel_size": cfg.get("ple_conv_kernel_size"),
        "hidden_size": cfg.get("hidden_size"),
        "hc_count": cfg.get("hc_count"),
        "vocab_size": cfg.get("vocab_size"),
        "layer_multipliers_i64": multipliers,
        "shard_rows": shard_shape[0] if shard_shape else None,
        "shard_width": shard_shape[1] if shard_shape else None,
        "shard_dtype": shard_dtype,
        "total_rows": (shard_shape[0] if shard_shape else 0)
        * len(ngram_files),
        "nsides": {
            "ple_tensors": len(
                [t for t in wm if "ple." in t]
            ),
            "ngram_shard_tensors": len(ngram_files),
            "ple_file_count": len(ple_files),
            "shard_files": ngram_files,
        },
        "projection_shape": {},
    }

    # 读取 00005 中其余投影/规范化张量形状
    with open(d / "model-00005-of-00131.safetensors", "rb") as f:
        L = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(L))
    for k in [
        "layers.1.ple.key_proj.weight",
        "layers.1.ple.norm_key.weight",
        "layers.1.ple.norm_query.weight",
        "layers.1.ple.norm_conv.weight",
        "layers.1.ple.ple_embedding.ngram_embedding.weight_scale",
    ]:
        t = f"model.language_model.{k}"
        if t in header:
            spec["projection_shape"][t.split("layers.1.ple.")[1]] = {
                "dtype": header[t]["dtype"],
                "shape": header[t]["shape"],
            }

    text = json.dumps(spec, indent=2, ensure_ascii=False)
    if args.out:
        args.out.write_text(text + "\n")
        print(f"spec written: {args.out}", file=sys.stderr)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
