#!/usr/bin/env python3
"""Inspect a Qwen-style model for real PLE/Engram table attributes.

This is a pure-metadata discovery tool.  It does not load weights, so it can run
quickly even on very large models.  It reads ``config.json`` and
``model.safetensors.index.json`` and prints the PLE-related layout that an
EngramDB disk adapter needs.

Example:

    python scripts/inspect_ple_attributes.py \
        --model /Volumes/My\ Passport/qwen38-ple
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def find_ple_keys(weight_map: dict[str, str]) -> list[str]:
    return [k for k in weight_map if ".ple." in k.lower()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()

    root = Path(args.model)
    config_path = root / "config.json"
    index_path = root / "model.safetensors.index.json"
    if not config_path.exists():
        raise SystemExit(f"missing config: {config_path}")
    if not index_path.exists():
        raise SystemExit(f"missing safetensors index: {index_path}")

    config = json.loads(config_path.read_text())
    text_cfg = config.get("text_config", config)

    print(f"model: {root}")
    print(f"architecture: {config.get('architectures')}")
    print(f"model_type: {config.get('model_type')} / text: {text_cfg.get('model_type')}")

    ple_attrs = {
        "ple_layer_ids": text_cfg.get("ple_layer_ids"),
        "ple_embed_dim": text_cfg.get("ple_embed_dim"),
        "ple_conv_kernel_size": text_cfg.get("ple_conv_kernel_size"),
        "ngram_size": text_cfg.get("ngram_size"),
        "ngram_vocab_size_base": text_cfg.get("ngram_vocab_size_base"),
        "split_ngram_parts": text_cfg.get("split_ngram_parts"),
        "heads_per_ngram": text_cfg.get("heads_per_ngram"),
        "make_ngram_vocab_size_divisible_by": text_cfg.get(
            "make_ngram_vocab_size_divisible_by"
        ),
    }
    for name, val in ple_attrs.items():
        print(f"  {name}: {val}")

    index = json.loads(index_path.read_text())
    weight_map = index["weight_map"]
    ple_keys = find_ple_keys(weight_map)
    print(f"PLE weight keys: {len(ple_keys)}")

    shard_keys = [k for k in ple_keys if "ngram_embedding.shard_" in k]
    shard_nums: set[int] = set()
    for k in shard_keys:
        part = k.split("ngram_embedding.shard_", 1)[1].split(".", 1)[0]
        try:
            shard_nums.add(int(part))
        except ValueError:
            pass
    print(f"  ngram_embedding shards: {len(shard_nums)}")
    if shard_nums:
        print(f"  shard index range: {min(shard_nums)}..{max(shard_nums)}")

    # Report the canonical PLE table path and its files.
    for k in sorted(ple_keys):
        if "ngram_embedding.shard_0." in k or k.endswith("layer_multipliers") or k.endswith("weight_scale"):
            print(f"  {k} -> {weight_map[k]}")

    if not ple_keys:
        print("NOTE: this model does not expose a PLE/Engram n-gram embedding in the safetensors index.")


if __name__ == "__main__":
    main()
