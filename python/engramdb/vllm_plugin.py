"""vLLM-facing PLE disk-offload plugin prototype.

This module is intentionally a thin prototype: it gives a future vLLM patch a
ready-made ``nn.Embedding`` replacement that fetches rows through EngramDB's
dedup + batch ``PleDiskGather``.  It deliberately does not import vLLM itself;
the integration point is the model attribute that holds the PLE table (for
example ``embed_tokens_per_layer`` on Qwen/Gemma-style models).
"""

from __future__ import annotations

from typing import Any

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]

from . import Store
from .vllm import PleDiskGather


class DiskPleEmbedding(nn.Module if nn is not None else object):  # type: ignore[misc]
    """Drop-in embedding module backed by an EngramDB Store.

    The constructor mirrors ``torch.nn.Embedding`` enough for typical PLE table
    replacement: ``num_embeddings`` is kept for shape/debugging while the actual
    rows live on disk.
    """

    def __init__(
        self,
        store: Store,
        num_embeddings: int,
        embedding_dim: int,
        dtype: Any = torch.float32 if torch is not None else None,
        cache_size: int = 4096,
    ) -> None:
        if nn is None:
            raise ImportError("DiskPleEmbedding requires PyTorch")
        super().__init__()
        self.store = store
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.dtype = dtype
        self.gather = PleDiskGather(store, row_bytes=self.embedding_dim * self.dtype.itemsize)

    def forward(self, indices: Any) -> Any:
        if torch is None:
            raise RuntimeError("DiskPleEmbedding.forward requires PyTorch")
        flat = indices.reshape(-1).cpu().tolist()
        raw = self.gather.fetch(flat)
        expected = int(indices.numel()) * self.embedding_dim
        if len(raw) != expected * self.dtype.itemsize:
            raise RuntimeError(
                f"EngramDB fetch returned {len(raw)} bytes, expected "
                f"{expected * self.dtype.itemsize}"
            )
        data = torch.frombuffer(bytearray(raw), dtype=self.dtype)
        return data.reshape(*indices.shape, self.embedding_dim)


def patch_named_embedding(
    module: Any,
    attr_name: str,
    store: Store,
    embedding_dim: int,
    dtype: Any = None,
    cache_size: int = 4096,
) -> DiskPleEmbedding:
    """Replace a named ``nn.Embedding`` attribute with a disk-backed one.

    ``attr_name`` may be a dotted path, e.g. ``model.embed_tokens_per_layer``.
    If ``dtype`` is omitted, the original embedding's weight dtype is used.
    """
    if nn is None:
        raise ImportError("patch_named_embedding requires PyTorch")

    parent_path, _, leaf = attr_name.rpartition(".")
    parent = module.get_submodule(parent_path) if parent_path else module
    old = getattr(parent, leaf)
    num_embeddings = getattr(old, "num_embeddings", None)
    if num_embeddings is None:
        num_embeddings = getattr(old, "weight", None).shape[0]
    if dtype is None:
        dtype = getattr(old, "weight", None).dtype

    new = DiskPleEmbedding(
        store=store,
        num_embeddings=int(num_embeddings),
        embedding_dim=int(embedding_dim),
        dtype=dtype,
        cache_size=cache_size,
    )
    setattr(parent, leaf, new)
    return new
