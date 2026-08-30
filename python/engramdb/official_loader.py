"""Helpers for loading an official Qwen4Exp-style model with disk-backed PLE.

This module is the "Phase B" bridge between HuggingFace official model classes
and the EngramDB disk PLE adapter.  It intentionally does not import
``transformers`` at module import time, so the package stays lightweight; all
heavy/optional dependencies are imported by the caller.
"""

from __future__ import annotations

from typing import Any

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


def filter_ngram_shard_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``state_dict`` with PLE n-gram embedding weights removed.

    The official Qwen4-Exp checkpoint stores PLE rows as
    ``...ngram_embedding.shard_N.weight`` and/or ``...ngram_embedding.weight``.
    EngramDB does not load these into RAM; they are served from disk instead.
    """
    filtered: dict[str, Any] = {}
    for key, value in state_dict.items():
        if "ngram_embedding" in key:
            continue
        filtered[key] = value
    return filtered


def _is_official_ngram_module(module: Any) -> bool:
    cls = type(module).__name__
    return cls == "Qwen4ExpTextNGramEmbedding" or cls.endswith("NGramEmbedding")


def _resolve_scale(info: dict[str, Any] | None, scale: float | None) -> float:
    if scale is not None:
        return float(scale)
    if info is not None and info.get("weight_scale") is not None:
        return float(info["weight_scale"])
    return 1.0


def install_disk_ple_in_official_model(
    model: Any,
    store: Any,
    info: dict[str, Any] | None = None,
    model_dir: str | None = None,
    scale: float | None = None,
    cache_size: int = 4096,
    layer_ids: list[int] | None = None,
) -> list[str]:
    """Replace every official PLE ``ple_embedding`` with a disk-backed adapter.

    Returns the list of replaced module paths.
    """
    if torch is None or nn is None:
        raise ImportError("install_disk_ple_in_official_model requires PyTorch")
    if info is None and model_dir is not None:
        from .ple_discovery import discover_ple

        info = discover_ple(model_dir)
    if info is None:
        raise ValueError(
            "install_disk_ple_in_official_model requires `info` or `model_dir`"
        )

    from .ple_adapter import DiskPleNGramEmbedding

    embedding_dim = int(info["ple_embed_dim"])
    multipliers = info.get("layer_multipliers") or info.get("rowid_multipliers")
    scale = _resolve_scale(info, scale)

    replaced: list[str] = []
    layer_filter = set(layer_ids) if layer_ids is not None else None
    for path, module in list(model.named_modules()):
        if not _is_official_ngram_module(module):
            continue
        # The official class is usually mounted as `.ple_embedding`; if it is
        # mounted directly somewhere else we still replace it.
        if not path.endswith(".ple_embedding") and "ple.ple_embedding" not in path:
            # Only replace paths that clearly belong to a PLE layer.
            if "ngram_embedding" not in path and "ple" not in path:
                continue
        # Extract layer id when possible, e.g.
        # model.language_model.layers.1.ple.ple_embedding -> 1
        layer_id: int | None = None
        for part in path.split("."):
            if part.isdigit():
                layer_id = int(part)
                break
        if layer_filter is not None and layer_id not in layer_filter:
            continue

        parent_path, _, leaf = path.rpartition(".")
        parent = model.get_submodule(parent_path) if parent_path else model
        disk = DiskPleNGramEmbedding(
            store=store,
            embedding_dim=embedding_dim,
            num_heads=16,
            layer_multipliers=multipliers,
            scale=scale,
            cache_size=cache_size,
        )
        setattr(parent, leaf, disk)
        replaced.append(path)

    if not replaced:
        raise RuntimeError(
            "no official Qwen4ExpTextNGramEmbedding modules found in the model"
        )
    return replaced


def load_state_dict_without_ngram_shards(
    model: Any,
    state_dict: dict[str, Any],
    strict: bool = False,
) -> Any:
    """Load a checkpoint into the model while skipping PLE ngram rows."""
    return model.load_state_dict(filter_ngram_shard_state_dict(state_dict), strict=strict)
