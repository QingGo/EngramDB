
"""Minimal vLLM-oriented disk gather helper.

This module is intentionally engine-agnostic: it provides the same primitives
that a vLLM PLE mmap patch would need (dedup, batched disk fetch, expansion),
so a future vLLM plugin can use EngramDB instead of raw ``np.memmap`` gathers.
"""

from __future__ import annotations

from typing import Any, Iterable

from . import Store


def fetch_e_t_tensor(
    store: Store,
    rowids: Iterable[int],
    scale: float = 1.0,
    num_heads: int = 16,
    head_dim: int = 160,
    dtype: Any = None,
    out_dtype: Any = None,
    dedup: bool = False,
) -> Any:
    """Directly fetch FP8 PLE rows and return a float torch tensor.

    This is the fast path for training/precompute: it bypasses the Python
    per-row ``bytes`` slicing and ``b"".join`` that made ``PleDiskGather.fetch``
    slow on tens of thousands of rows.  The only Python work is converting the
    rowid iterable to a list (and optionally deduplicating); the actual data is
    returned as one contiguous byte buffer from ``Store.fetch`` and converted
    in torch.

    Args:
        store: Open EngramDB Store.
        rowids: Flat row ids in the final access order.
        scale: FP8 dequant scale applied after conversion.
        num_heads: Number of n-gram heads (default Qwen PLE 16).
        head_dim: Bytes / feature width per head (default Qwen PLE 160).
        dtype: Raw dtype in the store (default ``torch.float8_e4m3fn``).
        out_dtype: Output dtype (default ``torch.float32``).
        dedup: When true, read only unique rows and expand back in torch.

    Returns:
        Tensor of shape ``[len(rowids) // num_heads, num_heads, head_dim]``
        (or an empty tensor of that shape when ``rowids`` is empty).
    """
    import torch

    if dtype is None:
        dtype = torch.float8_e4m3fn
    if out_dtype is None:
        out_dtype = torch.float32

    rowids_list = list(rowids)
    n_rows = len(rowids_list)
    n_heads = int(num_heads or 1)
    if n_rows == 0:
        return torch.empty((0, n_heads, head_dim), dtype=out_dtype)
    if n_rows % n_heads != 0:
        raise ValueError(
            f"rowids length {n_rows} is not divisible by num_heads={n_heads}"
        )
    n_tokens = n_rows // n_heads

    if dedup:
        seen: set[int] = set()
        unique: list[int] = []
        for r in rowids_list:
            if r not in seen:
                seen.add(r)
                unique.append(r)
        raw = store.fetch(unique)
        batch = torch.frombuffer(bytearray(raw), dtype=dtype).to(out_dtype)
        if head_dim is not None:
            batch = batch.reshape(len(unique), head_dim)
        index_of = {r: i for i, r in enumerate(unique)}
        idx = torch.tensor([index_of[r] for r in rowids_list], dtype=torch.long)
        arr = batch[idx]
    else:
        raw = store.fetch(rowids_list)
        arr = torch.frombuffer(bytearray(raw), dtype=dtype)

    arr = arr.to(out_dtype)
    if scale != 1.0:
        arr = arr * scale
    if head_dim is not None:
        arr = arr.reshape(n_tokens, n_heads, head_dim)
    return arr


class PleDiskGather:
    """Dedup + EngramDB batch fetch + expand to original row order."""

    def __init__(self, store: Store, row_bytes: int):
        self.store = store
        self.row_bytes = row_bytes

    def fetch(self, rowids: Iterable[int]) -> bytes:
        rowids_list = list(rowids)
        if not rowids_list:
            return b""

        # Fast path: return the contiguous Store.fetch buffer directly.  The
        # rowids are already in the requested access order, so no Python
        # dedup/slice/join is needed.  Use `fetch_unique` when the caller only
        # wants unique rows or wants to deduplicate before expanding manually.
        raw = self.store.fetch(rowids_list)
        if len(raw) != len(rowids_list) * self.row_bytes:
            raise RuntimeError(
                f"EngramDB fetch returned {len(raw)} bytes for "
                f"{len(rowids_list)} rows x {self.row_bytes}"
            )
        return raw

    def fetch_tensor(
        self,
        rowids: Iterable[int],
        scale: float = 1.0,
        num_heads: int = 16,
        head_dim: int | None = None,
        dtype: Any = None,
        out_dtype: Any = None,
        dedup: bool = False,
    ) -> Any:
        """Directly return a torch tensor, bypassing Python byte-slice expansion."""
        if head_dim is None:
            head_dim = self.row_bytes
        return fetch_e_t_tensor(
            self.store,
            rowids,
            scale=scale,
            num_heads=num_heads,
            head_dim=head_dim,
            dtype=dtype,
            out_dtype=out_dtype,
            dedup=dedup,
        )

    def fetch_unique(self, rowids: Iterable[int]) -> bytes:
        """Return only unique rows in first-seen order (for staging buffers)."""
        seen: set[int] = set()
        unique: list[int] = []
        for r in rowids:
            if r not in seen:
                seen.add(r)
                unique.append(r)
        return self.store.fetch(unique)
