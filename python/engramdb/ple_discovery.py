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
import struct
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

    weight_scale = None
    try:
        weight_scale = load_ple_weight_scale(root)
    except Exception:
        weight_scale = None

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
        "weight_scale": weight_scale,
        "example_ple_keys": ple_keys[:8],
    }


def _bf16_to_float(bits: int) -> float:
    sign = (bits >> 15) & 1
    exp = (bits >> 7) & 0xff
    mant = bits & 0x7f
    if exp == 0:
        v = mant * (2.0 ** -7) * (2.0 ** -126)
    elif exp == 0xff:
        v = float("inf") if mant == 0 else float("nan")
    else:
        v = (1.0 + mant / 128.0) * (2.0 ** (exp - 127))
    return -v if sign else v


def _f16_to_float(bits: int) -> float:
    sign = (bits >> 15) & 1
    exp = (bits >> 10) & 0x1f
    mant = bits & 0x3ff
    if exp == 0:
        v = mant * (2.0 ** -10) * (2.0 ** -14)
    elif exp == 0x1f:
        v = float("inf") if mant == 0 else float("nan")
    else:
        v = (1.0 + mant / 1024.0) * (2.0 ** (exp - 15))
    return -v if sign else v


def read_safetensors_scalar(path: str | Path, tensor_name: str) -> float:
    """Read a small scalar tensor from a safetensors file without loading it."""
    path = Path(path)
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
        entry = header.get(tensor_name)
        if entry is None:
            raise KeyError(f"tensor {tensor_name!r} not in {path.name}")
        start, end = entry["data_offsets"]
        dtype = entry.get("dtype", "F32")
        f.seek(8 + header_len + start)
        raw = f.read(end - start)
    if dtype == "F32":
        if len(raw) != 4:
            raise ValueError(f"expected 4 bytes for F32 scalar, got {len(raw)}")
        return struct.unpack("<f", raw)[0]
    if dtype in ("BF16", "F16"):
        if len(raw) != 2:
            raise ValueError(f"expected 2 bytes for {dtype} scalar, got {len(raw)}")
        u = struct.unpack("<H", raw)[0]
        return _bf16_to_float(u) if dtype == "BF16" else _f16_to_float(u)
    raise NotImplementedError(f"cannot read scalar dtype {dtype!r}")


def load_ple_weight_scale(
    model_dir: str | Path,
    tensor_name: str | None = None,
) -> float:
    """Read the real PLE FP8 ``weight_scale`` from a Qwen checkpoint.

    The returned value is exactly what DPE FP8 adapters need to dequantize
    stored 160-byte PLE rows.
    """
    root = Path(model_dir)
    index_path = root / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"missing safetensors index: {index_path}")
    index = json.loads(index_path.read_text())
    weight_map = index["weight_map"]
    if tensor_name is not None:
        key = tensor_name
    else:
        key = next(
            (
                k for k in weight_map
                if k.endswith(".ngram_embedding.weight_scale")
                or k.endswith("ngram_embedding.weight_scale")
            ),
            None,
        )
    if key is None:
        raise KeyError("no PLE ngram_embedding.weight_scale in checkpoint index")
    return read_safetensors_scalar(root / weight_map[key], key)
