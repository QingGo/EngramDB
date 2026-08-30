"""vLLM-facing PLE disk-offload plugin prototype.

This module is intentionally a thin prototype: it gives a future vLLM patch a
ready-made ``nn.Embedding`` replacement that fetches rows through EngramDB's
dedup + batch ``PleDiskGather``.  It deliberately does not import vLLM itself;
the integration point is the model attribute that holds the PLE table (for
example ``embed_tokens_per_layer`` on Qwen/Gemma-style models).
"""

from __future__ import annotations

from collections import OrderedDict
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
        self.row_bytes = self.embedding_dim * self.dtype.itemsize
        self.cache_size = int(cache_size)
        self.gather = PleDiskGather(store, row_bytes=self.row_bytes)
        self._cache: OrderedDict[int, bytes] = OrderedDict()

    def _get_missing(self, flat: list[int]) -> dict[int, bytes]:
        """Fetch uncached rows in batch and return a rowid -> bytes map.

        When ``cache_size`` is positive, the rows are also stored in the LRU
        cache.  A cache size of zero intentionally disables caching, which is
        useful for raw-disk A/B benchmarks.
        """
        missing: list[int] = []
        seen: set[int] = set()
        for r in flat:
            if r not in self._cache and r not in seen:
                seen.add(r)
                missing.append(r)
        if not missing:
            return {}
        raw = self.store.fetch(missing)
        expected = len(missing) * self.row_bytes
        if len(raw) != expected:
            raise RuntimeError(
                f"EngramDB fetch returned {len(raw)} bytes, expected {expected}"
            )
        fetched: dict[int, bytes] = {}
        for i, rowid in enumerate(missing):
            fetched[rowid] = raw[i * self.row_bytes:(i + 1) * self.row_bytes]
        if self.cache_size > 0:
            self._cache.update(fetched)
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return fetched

    def forward(self, indices: Any) -> Any:
        if torch is None:
            raise RuntimeError("DiskPleEmbedding.forward requires PyTorch")
        flat = indices.reshape(-1).cpu().tolist()
        fetched = self._get_missing(flat)
        if self.cache_size <= 0:
            # Raw no-cache path: every requested row is in `fetched` because the
            # cache is intentionally empty on every call.
            raw = b"".join(fetched[r] for r in flat)
        else:
            raw = b"".join(self._cache[r] for r in flat)
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



def patch_model_class_ple(
    model_class: type,
    store: Store,
    attr_name: str,
    embedding_dim: int,
    dtype: Any = None,
    cache_size: int = 4096,
) -> type:
    """Patch a model *class* so every instance uses EngramDB for its PLE table.

    This is the no-source-change entry point for vLLM/SGLang-style serving:
    call it before constructing the engine/model, with the model class that owns
    the PLE embedding attribute.  The class ``__init__`` is wrapped and, after
    normal construction, the named PLE embedding is replaced on the instance.

    Example::

        from engramdb.vllm_plugin import patch_model_class_ple
        patch_model_class_ple(
            Qwen3_8FlashNextNGramEmbedding,
            store=store,
            attr_name="embed_tokens_per_layer",
            embedding_dim=hidden_size_per_layer_input,
        )
        llm = LLM(model=..., ...)
    """
    if not hasattr(model_class, "__init__"):
        raise TypeError(f"{model_class!r} is not a normal class with __init__")
    if getattr(model_class, "_engramdb_ple_patched", False):
        return model_class

    original_init = model_class.__init__

    def patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        patch_named_embedding(
            self,
            attr_name=attr_name,
            store=store,
            embedding_dim=embedding_dim,
            dtype=dtype,
            cache_size=cache_size,
        )

    model_class.__init__ = patched_init  # type: ignore[method-assign]
    model_class._engramdb_ple_patched = True  # type: ignore[attr-defined]
    return model_class


# vLLM-specific alias; SGLang can reuse the same generic helper via engramdb.sglang.
install_vllm_ple = patch_model_class_ple
