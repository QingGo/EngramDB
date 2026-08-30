"""Discovery of Qwen-style PLE/Engram table metadata from a model directory.

This module is intentionally pure-Python and does not load model weights.  It
reads HuggingFace ``config.json`` and ``model.safetensors.index.json`` metadata
to find:

* whether the model has a PLE/Engram n-gram embedding
* the canonical attribute path of the PLE table
* the number of n-gram embedding shards
* relevant PLE configuration values such as ``ple_embed_dim``,
  ``ngram_size``, ``split_ngram_parts``, and ``ngram_vocab_size_base``

Example::

    from engramdb.ple_discovery import discover_ple

    info = discover_ple("/path/to/Qwen3.8-Flash-Next")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _text_config(config: dict[str, Any]) -> dict[str, Any]:
    if isinstance(config.get("text_config"), dict):
        return config["text_config"]
    return config


def discover_ple(model_dir: str | Path) -> dict[str, Any] | None:
    """Return PLE metadata for a model directory, or None when no PLE exists."""
    root = Path(model_dir)
    config_path = root / "config.json"
    index_path = root / "model.safetensors.index.json"
    if not config_path.exists():
        raise FileNotFoundError(f"missing config.json: {config_path}")
    if not index_path.exists():
        raise FileNotFoundError(f"missing model.safetensors.index.json: {index_path}")

    config = json.loads(config_path.read_text())
    text_cfg = _text_config(config)

    ple_layer_ids = text_cfg.get("ple_layer_ids")
    ple_keys: list[str] = []
    shard_keys: list[str] = []
    shard_count = 0

    if ple_layer_ids is not None or any(
        key in text_cfg for key in ("ple_embed_dim", "ngram_size", "split_ngram_parts")
    ):
        index = json.loads(index_path.read_text())
        weight_map = index["weight_map"]
        ple_keys = [k for k in weight_map if ".ple." in k.lower()]
        shard_keys = [k for k in ple_keys if "ngram_embedding.shard_" in k]
        shard_count = len(set(
            k.split("ngram_embedding.shard_", 1)[1].split(".", 1)[0]
            for k in shard_keys
        ))

    if not ple_keys and ple_layer_ids is None:
        return None

    # The first table shard is the canonical example; deeper adapters may use
    # this path pattern to construct rowid-to-shard mappings.
    shard_pattern = None
    if shard_keys:
        base = shard_keys[0].split("ngram_embedding.shard_", 1)[0]
        shard_pattern = base + "ngram_embedding.shard_{shard}.weight"

    return {
        "architecture": config.get("architectures"),
        "model_type": config.get("model_type"),
        "text_model_type": text_cfg.get("model_type"),
        "ple_layer_ids": ple_layer_ids,
        "ple_embed_dim": text_cfg.get("ple_embed_dim"),
        "ple_conv_kernel_size": text_cfg.get("ple_conv_kernel_size"),
        "ngram_size": text_cfg.get("ngram_size"),
        "ngram_vocab_size_base": text_cfg.get("ngram_vocab_size_base"),
        "split_ngram_parts": text_cfg.get("split_ngram_parts"),
        "heads_per_ngram": text_cfg.get("heads_per_ngram"),
        "make_ngram_vocab_size_divisible_by": text_cfg.get(
            "make_ngram_vocab_size_divisible_by"
        ),
        "ple_weight_key_count": len(ple_keys),
        "ngram_embedding_shard_count": shard_count,
        "ple_shard_pattern": shard_pattern,
        "example_ple_keys": ple_keys[:8],
    }
