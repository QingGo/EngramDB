
"""Optional integrations for EngramDB.

The main module intentionally stays dependency-light.  This submodule pulls in
PyTorch (and optionally engram-peft) only when the user wants to replace
`MultiHeadEmbedding` with an EngramDB-backed disk store.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import torch
from torch import nn

from . import Store


class DiskMultiHeadEmbedding(nn.Module):
    """Drop-in replacement for engram_peft.layer.MultiHeadEmbedding.

    Reads embedding rows from an EngramDB Store instead of an in-memory
    `nn.Embedding`.  Includes a small LRU cache.
    """

    def __init__(
        self,
        primes: list[int],
        embedding_dim_per_head: int,
        store: Store,
        dtype: Any = torch.float32,
        cache_size: int = 4096,
        scale: float = 1.0,
        output_dtype: Any | None = None,
        **kwargs: Any,
    ):
        super().__init__()
        self.embedding_dim_per_head = embedding_dim_per_head
        offsets = [0]
        for p in primes[:-1]:
            offsets.append(offsets[-1] + p)
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long))
        self.store = store
        self.dtype = dtype
        self.scale = float(scale)
        self.output_dtype = output_dtype
        self._sparse = False
        self._cache: OrderedDict[int, bytes] = OrderedDict()
        self._cache_size = max(0, int(cache_size))

    def forward(self, hash_indices):
        shifted = hash_indices.to(self.offsets.device) + self.offsets
        flat = shifted.reshape(-1).cpu().tolist()
        row_len = self.embedding_dim_per_head * self.dtype.itemsize

        misses: list[int] = []
        for i in flat:
            if i not in self._cache:
                misses.append(i)
        if misses:
            raw = self.store.fetch(misses)
            if len(raw) != len(misses) * row_len:
                raise RuntimeError(
                    f"disk fetch returned {len(raw)} bytes for {len(misses)} rows"
                )
            for j, idx in enumerate(misses):
                self._cache[idx] = raw[j * row_len:(j + 1) * row_len]
                if self._cache_size > 0:
                    while len(self._cache) > self._cache_size:
                        self._cache.popitem(last=False)

        raw = b"".join(self._cache[i] for i in flat)
        expected = int(shifted.numel()) * self.embedding_dim_per_head
        if len(raw) != expected * self.dtype.itemsize:
            raise RuntimeError(
                f"disk fetch returned {len(raw)} bytes, expected {expected * self.dtype.itemsize}"
            )
        data = torch.frombuffer(bytearray(raw), dtype=self.dtype)
        if self.scale != 1.0 or self.output_dtype is not None:
            out_dtype = self.output_dtype or torch.float32
            data = data.to(out_dtype) * self.scale
        return data.reshape(*shifted.shape, self.embedding_dim_per_head)


def install_disk_multi_head_embedding(
    store: Store,
    cache_size: int = 4096,
    dtype: Any = torch.float32,
    scale: float = 1.0,
    output_dtype: Any | None = None,
) -> None:
    """Patch engram_peft's MultiHeadEmbedding to use an EngramDB Store.

    For FP8 PLE tables pass ``dtype=torch.float8_e4m3fn`` and the real
    ``weight_scale`` as ``scale``; the module will dequantize to float32
    (or ``output_dtype``) before returning.

    Call this before constructing the Engram model, e.g. after
    ``uv add engramdb-python`` and before ``get_engram_model``.
    """
    import engram_peft.layer as layer

    original = layer.MultiHeadEmbedding
    layer._original_multi_head_embedding = original

    class PatchedDiskMultiHeadEmbedding(DiskMultiHeadEmbedding):
        def __init__(self, primes, embedding_dim_per_head, sparse=True, **kwargs):
            super().__init__(
                primes,
                embedding_dim_per_head,
                store=store,
                cache_size=cache_size,
                dtype=dtype,
                scale=scale,
                output_dtype=output_dtype,
                **kwargs,
            )

    PatchedDiskMultiHeadEmbedding.__name__ = "DiskMultiHeadEmbedding"
    layer.MultiHeadEmbedding = PatchedDiskMultiHeadEmbedding


def install_real_qwen_ple_embedding(
    store: Store,
    scale: float = 0.0002,
    cache_size: int = 4096,
) -> None:
    """Patch engram_peft for real Qwen PLE FP8 Store-I rows.

    This is the convenience wrapper for the Qwen3.8-Flash-Next / Qwen4Exp PLE
    table: 160-byte FP8 rows with a global ``weight_scale``.  The patched
    MultiHeadEmbedding reads FP8, dequantizes with ``scale``, and returns
    float32 (which downstream projections can cast as needed).
    """
    import torch as _torch

    return install_disk_multi_head_embedding(
        store,
        cache_size=cache_size,
        dtype=_torch.float8_e4m3fn,
        scale=scale,
        output_dtype=_torch.float32,
    )
