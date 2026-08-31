"""Helpers for loading an official Qwen4Exp-style model with disk-backed PLE.

This module is the "Phase B" bridge between HuggingFace official model classes
and the EngramDB disk PLE adapter.  It intentionally does not import
``transformers`` at module import time, so the package stays lightweight; all
heavy/optional dependencies are imported by the caller.
"""

from __future__ import annotations

import inspect
from contextlib import contextmanager
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Iterator

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


def _resolve_num_heads(info: dict[str, Any] | None, fallback: int = 16) -> int:
    if info is None:
        return fallback
    ngram_size = info.get("ngram_size")
    heads_per_ngram = info.get("heads_per_ngram")
    if ngram_size is not None and heads_per_ngram is not None:
        return max(1, (int(ngram_size) - 1) * int(heads_per_ngram))
    return fallback


def _find_official_ngram_embedding_class() -> type | None:
    """Locate the official Qwen4-Exp PLE n-gram embedding class.

    The exact import path depends on the Transformers version.  This function
    tries the known module first and leaves the door open for callers to pass
    a class explicitly when a non-standard/customized modeling file is used.
    """
    try:
        from transformers.models.qwen4_exp.modeling_qwen4_exp import (  # noqa: F401
            Qwen4ExpTextNGramEmbedding as cls,
        )
        return cls
    except Exception:
        return None


@contextmanager
def patch_official_ngram_embedding_for_disk_load(
    embedding_class: type | None = None,
    placeholder_rows: int = 1,
) -> Iterator[None]:
    """Temporarily replace the giant PLE ``nn.Embedding`` with a tiny stub.

    The official ``Qwen4ExpTextNGramEmbedding`` constructor allocates an
    ``nn.Embedding`` whose row count is the full PLE vocabulary (hundreds of
    millions of rows).  During model construction we do not need that table:
    the PLE module is replaced with ``DiskPleNGramEmbedding`` immediately after
    loading the non-PLE checkpoint.  This context manager patches the
    constructor so the giant table is never allocated.

    Use it around ``AutoModelForCausalLM.from_config`` / ``from_pretrained``::

        with patch_official_ngram_embedding_for_disk_load():
            model = AutoModelForCausalLM.from_config(config)

    Args:
        embedding_class:
            Optional explicit class.  If omitted, the official Transformers
            ``Qwen4ExpTextNGramEmbedding`` is located automatically.
        placeholder_rows:
            Row count for the temporary placeholder embedding.  The default is
            one row, which is enough to keep the model object structurally valid
            while consuming negligible memory.
    """
    if torch is None or nn is None:
        raise ImportError("patch_official_ngram_embedding_for_disk_load requires PyTorch")
    if embedding_class is None:
        embedding_class = _find_official_ngram_embedding_class()
    if embedding_class is None:
        raise RuntimeError(
            "could not locate Qwen4ExpTextNGramEmbedding; pass `embedding_class` "
            "explicitly or use a Transformers build that ships Qwen4-Exp"
        )

    module = inspect.getmodule(embedding_class)
    original_init = embedding_class.__init__
    embedded_nn = getattr(module, "nn", None) if module is not None else None
    if embedded_nn is None:
        embedded_nn = getattr(original_init, "__globals__", {}).get("nn")
    if embedded_nn is None or not hasattr(embedded_nn, "Embedding"):
        raise RuntimeError(
            f"cannot patch {embedding_class!r}: its module has no nn.Embedding"
        )

    original_embedding = embedded_nn.Embedding
    requested_rows = max(1, int(placeholder_rows))

    class _PlaceholderEmbedding(original_embedding):  # type: ignore[misc, valid-type]
        def __init__(self, num_embeddings: int, embedding_dim: int, *args: Any, **kwargs: Any) -> None:
            # Record the requested size for diagnostics, but keep only a tiny
            # parameter so the model can be constructed and loaded without the
            # multi-hundred-GB PLE row table.
            self._requested_num_embeddings = int(num_embeddings)
            actual_rows = min(int(num_embeddings), requested_rows)
            super().__init__(actual_rows, embedding_dim, *args, **kwargs)

    def _init_with_placeholder(self, *args: Any, **kwargs: Any) -> None:
        embedded_nn.Embedding = _PlaceholderEmbedding
        try:
            original_init(self, *args, **kwargs)
        finally:
            embedded_nn.Embedding = original_embedding

    # Install the wrapper for the duration of the context.
    embedding_class.__init__ = _init_with_placeholder  # type: ignore[method-assign]
    try:
        yield
    finally:
        embedding_class.__init__ = original_init  # type: ignore[method-assign]


def install_disk_ple_prefetch_hook(model: Any) -> Any:
    """Install a model-level pre-hook that prefetches all disk PLE rows.

    PLE rowids depend only on ``input_ids``, so calling ``prefetch`` before the
    first transformer layer lets the disk I/O overlap with earlier-layer compute.
    """
    from .ple_adapter import DiskPleNGramEmbedding

    disk_modules = [
        m for m in model.modules() if isinstance(m, DiskPleNGramEmbedding)
    ]
    if not disk_modules:
        raise RuntimeError(
            "install_disk_ple_prefetch_hook requires at least one DiskPleNGramEmbedding"
        )

    def _prefetch_hook(module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        del module
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]
        if input_ids is None:
            return None
        for ple in disk_modules:
            ple.prefetch(input_ids)
        return None

    return model.register_forward_pre_hook(_prefetch_hook)


def install_disk_ple_in_official_model(
    model: Any,
    store: Any,
    info: dict[str, Any] | None = None,
    model_dir: str | None = None,
    scale: float | None = None,
    cache_size: int = 4096,
    layer_ids: list[int] | None = None,
    prefetch: bool = False,
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

    from .ple_adapter import (
        DiskPleNGramEmbedding,
        head_offsets,
        head_vocab_sizes,
    )

    embedding_dim = int(info["ple_embed_dim"])
    multipliers = info.get("layer_multipliers") or info.get("rowid_multipliers")
    scale = _resolve_scale(info, scale)
    num_heads = _resolve_num_heads(info)
    ngram_base = int(info.get("ngram_vocab_size_base") or 20_000_000)
    divisor = int(info.get("make_ngram_vocab_size_divisible_by") or 128)
    prime_sizes = head_vocab_sizes(base=ngram_base, heads=num_heads)
    head_off = head_offsets(prime_sizes)

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
            num_heads=num_heads,
            layer_multipliers=multipliers,
            scale=scale,
            cache_size=cache_size,
            ngram_size=int(info.get("ngram_size", 3)),
            heads_per_ngram=int(info.get("heads_per_ngram", 8)),
            prime_sizes=prime_sizes,
            offsets=head_off,
            divisor=divisor,
        )
        setattr(parent, leaf, disk)
        replaced.append(path)

    if not replaced:
        raise RuntimeError(
            "no official Qwen4ExpTextNGramEmbedding modules found in the model"
        )
    if prefetch:
        install_disk_ple_prefetch_hook(model)
    return replaced


def load_state_dict_without_ngram_shards(
    model: Any,
    state_dict: dict[str, Any],
    strict: bool = False,
) -> Any:
    """Load a checkpoint into the model while skipping PLE ngram rows."""
    return model.load_state_dict(filter_ngram_shard_state_dict(state_dict), strict=strict)


@dataclass
class CheckpointLoadResult:
    """Result of loading a checkpoint while skipping PLE ngram rows."""

    missing_keys: list[str]
    unexpected_keys: list[str]
    skipped_ngram_tensors: int
    loaded_tensors: int


def load_official_checkpoint_without_ngram_shards(
    model: Any,
    model_dir: str | Path,
    strict: bool = False,
) -> CheckpointLoadResult:
    """Load all non-PLE sharded safetensors from a HuggingFace model directory.

    This streams through the safetensors shards with ``safe_open`` and only
    loads tensors whose keys do not contain ``ngram_embedding``.  The PLE shard
    rows (the multi-hundred-GB part of a Qwen4-Exp checkpoint) are never read
    into memory.
    """
    root = Path(model_dir)
    index_path = root / "model.safetensors.index.json"
    if index_path.exists():
        import json

        index = json.loads(index_path.read_text())
        shard_files = sorted(set(index["weight_map"].values()))
    else:
        shard_files = sorted(
            p.name
            for p in root.glob("*.safetensors")
            if not p.name.endswith(".index.json")
        )
    if not shard_files:
        raise FileNotFoundError(f"no safetensors shards found in {root}")

    try:
        from safetensors import safe_open
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "load_official_checkpoint_without_ngram_shards requires `safetensors`"
        ) from exc

    filtered: dict[str, Any] = {}
    skipped = 0
    for shard in shard_files:
        path = root / shard
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if "ngram_embedding" in key:
                    skipped += 1
                    continue
                filtered[key] = f.get_tensor(key)
    result = model.load_state_dict(filtered, strict=strict)
    return CheckpointLoadResult(
        missing_keys=list(getattr(result, "missing_keys", [])),
        unexpected_keys=list(getattr(result, "unexpected_keys", [])),
        skipped_ngram_tensors=skipped,
        loaded_tensors=len(filtered),
    )

