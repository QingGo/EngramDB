
#!/usr/bin/env python3
"""Minimal engram-peft interop example using EngramDB's Python bridge.

This example shows how to make `MultiHeadEmbedding` read its embedding rows
from an EngramDB Store-I directory instead of an in-memory `nn.Embedding`.

It is intentionally minimal:
- It does not require a full model yet; it verifies the disk-backed
  MultiHeadEmbedding produces the same logical tensor as the in-memory version.
- Later the same class can be injected into `EngramLayer` before creating a
  model, which gives an end-to-end EngramDB-backed engram-peft path.

Requirements:
- `engramdb` Python package from this repo (ctypes bridge).
- `torch` and `numpy` for the optional live check.
"""

from __future__ import annotations

import os
import struct
import tempfile
from pathlib import Path

try:
    import torch
    from torch import nn
except ImportError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]

import engramdb


def write_flat_store(
    directory: str,
    rows: list[bytes],
    row_width: int,
) -> str:
    """Write raw fixed-size rows as shard_000.bin and return the directory."""
    Path(directory).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(directory, "shard_000.bin"), "wb") as f:
        for r in rows:
            assert len(r) == row_width
            f.write(r)
    return directory


class DiskMultiHeadEmbedding(nn.Module if nn is not None else object):
    """Drop-in replacement for engram_peft.layer.MultiHeadEmbedding.

    It keeps the same constructor contract (`primes`, `embedding_dim_per_head`)
    and reads rows from an EngramDB Store.  The first version is inference-only;
    autograd/training support will be added once the end-to-end path is stable.
    """

    def __init__(
        self,
        primes: list[int],
        embedding_dim_per_head: int,
        store: engramdb.Store | None = None,
        dtype=torch.float32,
        **kwargs,
    ):
        super().__init__()
        self.embedding_dim_per_head = embedding_dim_per_head
        offsets = [0]
        for p in primes[:-1]:
            offsets.append(offsets[-1] + p)
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long))
        self.store = store
        self.dtype = dtype
        self._sparse = False

    def forward(self, hash_indices):
        if self.store is None:
            raise RuntimeError("DiskMultiHeadEmbedding requires an EngramDB store")
        # Same logical index mapping as the in-memory MultiHeadEmbedding.
        shifted = hash_indices.to(self.offsets.device) + self.offsets
        flat = shifted.reshape(-1).cpu().tolist()
        raw = self.store.fetch(flat)
        expected = int(shifted.numel()) * self.embedding_dim_per_head
        if len(raw) != expected * self.dtype.itemsize:
            raise RuntimeError(
                f"disk fetch returned {len(raw)} bytes, expected {expected * self.dtype.itemsize}"
            )
        data = torch.frombuffer(bytearray(raw), dtype=self.dtype)
        return data.reshape(*shifted.shape, self.embedding_dim_per_head)


def install_disk_multi_head_embedding(store: engramdb.Store) -> None:
    """Patch engram_peft's class to use the disk-backed implementation.

    Call this before constructing the Engram model.  It replaces the class
    object used by `EngramLayer.__init__`.
    """
    import engram_peft.layer as layer

    original = layer.MultiHeadEmbedding

    class PatchedDiskMultiHeadEmbedding(DiskMultiHeadEmbedding):
        def __init__(self, primes, embedding_dim_per_head, sparse=True, **kwargs):
            super().__init__(
                primes,
                embedding_dim_per_head,
                store=store,
                sparse=sparse,
                **kwargs,
            )

    PatchedDiskMultiHeadEmbedding.__name__ = "DiskMultiHeadEmbedding"
    layer.MultiHeadEmbedding = PatchedDiskMultiHeadEmbedding
    # Keep a reference so callers can restore easily.
    layer._original_multi_head_embedding = original


def build_demo_store(num_rows: int, row_width: int) -> tuple[engramdb.Store, str]:
    """Create a tiny random flat store for tests."""
    directory = tempfile.mkdtemp(prefix="engramdb-interop-")
    rows: list[bytes] = []
    for i in range(num_rows):
        rows.append(bytes((j + i) % 251 for j in range(row_width)))
    write_flat_store(directory, rows, row_width)
    return engramdb.Store(directory, 1, num_rows, row_width), directory


def self_check() -> None:
    if torch is None:
        print("torch not installed; skipping live self-check")
        print("The example code is still usable once deps are available.")
        return

    # Small table: 3 heads with small prime-like capacities.
    primes = [4, 5, 7]
    per_head = 3  # floats per row
    total = sum(primes)
    row_width = per_head * 4  # float32 bytes

    store, directory = build_demo_store(total, row_width)

    # Logical embedding weights as a flat tensor.
    table = torch.arange(total * per_head, dtype=torch.float32).reshape(total, per_head)
    # Rewrite the store so rows match the deterministic table.
    with open(os.path.join(directory, "shard_000.bin"), "wb") as f:
        for value in table.reshape(-1).tolist():
            f.write(struct.pack("<f", value))

    disk = DiskMultiHeadEmbedding(primes, per_head, store=store)
    # Same shape as engram-peft: [batch, seq, heads], each head picks a row.
    hash_indices = torch.tensor([[[0, 1, 2], [3, 4, 5]]])
    out = disk(hash_indices)
    # Compare with direct embedding indexing.
    offsets = torch.tensor([0, 4, 9])
    shifted = hash_indices + offsets
    expected = table[shifted.reshape(-1)].reshape(*shifted.shape, per_head)
    assert torch.equal(out, expected), "disk MultiHeadEmbedding mismatch"

    store.close()
    print("self_check OK: disk-backed MultiHeadEmbedding matches in-memory indexing")

def engram_layer_check() -> None:
    """Run a real EngramLayer forward with the disk-backed MultiHeadEmbedding.

    This exercises the actual engram_peft code path (EngramLayer -> gating ->
    short_conv) while the embedding rows are read from an EngramDB Store.
    """
    # Python 3.10/3.11 compatibility: `typing.override` is only in 3.12+.
    try:
        import typing
        import typing_extensions
        if not hasattr(typing, "override"):
            typing.override = typing_extensions.override
    except Exception:
        pass

    try:
        from engram_peft import EngramConfig, EngramLayer
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"engram_layer_check skipped (engram_peft unavailable: {exc})")
        return

    hidden_size = 32
    embedding_dim = 64
    ngram_sizes = [2, 3]
    n_head_per_ngram = 2
    total_heads = len(ngram_sizes) * n_head_per_ngram
    per_head = embedding_dim // total_heads
    primes = [4, 5, 7, 11]
    total = sum(primes)
    row_width = per_head * 4  # float32

    store, directory = build_demo_store(total, row_width)
    table = torch.arange(total * per_head, dtype=torch.float32).reshape(total, per_head)
    with open(os.path.join(directory, "shard_000.bin"), "wb") as f:
        for value in table.reshape(-1).tolist():
            f.write(struct.pack("<f", value))

    install_disk_multi_head_embedding(store)

    config = EngramConfig(
        hidden_size=hidden_size,
        embedding_dim=embedding_dim,
        ngram_sizes=ngram_sizes,
        n_head_per_ngram=n_head_per_ngram,
        target_layers=[0],
        engram_vocab_size_per_ngram=[20, 20],
        compressed_vocab_size=10,
        pad_id=0,
    )
    layer = EngramLayer(config=config, layer_id=0, primes=primes, compressor=None)

    hidden = torch.randn(2, 3, hidden_size)
    hashes = torch.tensor(
        [
            [[0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 0, 1]],
            [[3, 0, 1, 2], [0, 1, 2, 3], [1, 2, 3, 0]],
        ]
    )
    out = layer(hidden_states=hidden, engram_hash_indices=hashes)
    assert out.shape == hidden.shape, (out.shape, hidden.shape)
    store.close()
    print("engram_layer_check OK: real EngramLayer forward with disk-backed embeddings")



if __name__ == "__main__":
    self_check()
    engram_layer_check()
