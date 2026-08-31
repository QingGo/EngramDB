"""Disk-backed Qwen PLE n-gram embedding adapter.

This module provides a drop-in replacement for the inner ``ngram_embedding``
table of a Qwen4Exp-style PLE layer without materialising the full
multi-hundred-GB embedding weight.  The deterministic rowid generation follows
the official ``Qwen4ExpTextNGramEmbedding`` math; the actual row payloads are
read through EngramDB.

Example::

    from engramdb.ple_adapter import DiskPleNGramEmbedding

    ple = DiskPleNGramEmbedding(
        store=store,
        num_embeddings=padded_vocab_size,
        embedding_dim=160,
        num_heads=16,
        layer_multipliers=[...],
        scale=0.0002,
    )
"""

from __future__ import annotations

from typing import Any

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]

    class _DummyModule:
        pass

    nn = type("nn", (), {"Module": _DummyModule})  # type: ignore[assignment]

from .vllm_plugin import DiskPleEmbedding

PLE_EOS = 248044
PLE_NGRAM_SIZE = 3
PLE_HEADS_PER_NGRAM = 8
PLE_HEADS = 16
PLE_BASE = 20_000_000
PLE_DIVISOR = 128


def _is_prime(v: int) -> bool:
    if v < 2:
        return False
    if v % 2 == 0:
        return v == 2
    d = 3
    while d * d <= v:
        if v % d == 0:
            return False
        d += 2
    return True


def nth_prime_after(start: int, count: int) -> int:
    p = start
    for _ in range(count):
        p += 1
        while not _is_prime(p):
            p += 1
    return p


def head_vocab_sizes(base: int = PLE_BASE, heads: int = PLE_HEADS) -> list[int]:
    return [nth_prime_after(base - 1, i + 1) for i in range(heads)]


def head_offsets(sizes: list[int] | None = None) -> list[int]:
    sizes = sizes or head_vocab_sizes()
    offsets = [0]
    for s in sizes[:-1]:
        offsets.append(offsets[-1] + s)
    return offsets


def padded_vocab_size(
    sizes: list[int] | None = None,
    divisor: int = PLE_DIVISOR,
) -> int:
    sizes = sizes or head_vocab_sizes()
    total = sum(sizes)
    return (total + divisor - 1) // divisor * divisor


def disk_ple_from_discovery(
    store: Any,
    info: dict[str, Any],
    layer_multipliers: list[int] | None = None,
    scale: float | None = None,
    cache_size: int = 4096,
) -> "DiskPleNGramEmbedding":
    """Build a disk PLE adapter from metadata returned by ``discover_ple``.

    This is the automatic path: feed in the output of
    ``engramdb.ple_discovery.discover_ple(model_dir)`` and it derives the
    embedding width and number of n-gram heads from the real model config.
    If ``scale`` is omitted, the ``weight_scale`` from discovery is used
    automatically when available.
    """
    from .ple_discovery import discover_ple  # noqa: F401  (kept for discoverability)

    if not info:
        raise ValueError("PLE discovery returned no PLE metadata")
    if scale is None:
        scale = info.get("weight_scale")
        if scale is None:
            scale = 1.0
    if layer_multipliers is None:
        for key in ("layer_multipliers", "rowid_multipliers", "multipliers"):
            if info.get(key) is not None:
                layer_multipliers = [int(x) for x in info[key]]
                break
    ngram_heads = (int(info["ngram_size"]) - 1) * int(info["heads_per_ngram"])
    ngram_base = int(info.get("ngram_vocab_size_base") or PLE_BASE)
    divisor = int(info.get("make_ngram_vocab_size_divisible_by") or PLE_DIVISOR)
    prime_sizes = head_vocab_sizes(base=ngram_base, heads=ngram_heads)
    return DiskPleNGramEmbedding(
        store=store,
        embedding_dim=int(info["ple_embed_dim"]),
        num_heads=ngram_heads,
        layer_multipliers=layer_multipliers,
        scale=float(scale),
        cache_size=cache_size,
        ngram_size=int(info["ngram_size"]),
        heads_per_ngram=int(info["heads_per_ngram"]),
        prime_sizes=prime_sizes,
        offsets=head_offsets(prime_sizes),
        divisor=divisor,
    )


def _shift_right_ignore_eos(hist: list[int], shift: int, eos: int) -> list[int]:
    if shift == 0:
        return hist
    n = len(hist)
    prev_incl = [-1] * n
    last = -1
    for i, x in enumerate(hist):
        if x == eos:
            last = i
        prev_incl[i] = last
    out = []
    for i in range(n):
        seg_start = 0 if i == 0 else prev_incl[i - 1] + 1
        pos_in_seg = i - seg_start
        src = i - shift
        valid = pos_in_seg >= shift and src >= 0
        out.append(hist[src] if valid else eos)
    return out


def ple_rowids(
    tokens: list[int],
    multipliers: list[int],
    eos: int = PLE_EOS,
    sizes: list[int] | None = None,
    offsets: list[int] | None = None,
    ngram_size: int = PLE_NGRAM_SIZE,
    heads_per_ngram: int = PLE_HEADS_PER_NGRAM,
    history: list[int] | None = None,
) -> list[list[int]]:
    """Return ``[len(tokens), len(sizes)]`` rowids for a token sequence.

    When ``history`` is supplied it is used as the already-known n-gram context
    (for streaming/decode); otherwise the sequence is treated as starting after
    EOS padding.
    """
    sizes = sizes or head_vocab_sizes()
    offsets = offsets or head_offsets(sizes)
    if history is None:
        hist = [eos, eos] + list(tokens)
    else:
        hist = list(history) + list(tokens)
    shifted = [_shift_right_ignore_eos(hist, sh, eos) for sh in range(ngram_size)]
    ids_all = []
    for pos in range(len(hist)):
        row = []
        for ngram in range(2, ngram_size + 1):
            start = (ngram - 2) * heads_per_ngram
            end = start + heads_per_ngram
            mixed = shifted[0][pos] * multipliers[0]
            for order in range(1, ngram):
                mixed ^= shifted[order][pos] * multipliers[order]
            for h in range(start, end):
                rid = (mixed % sizes[h]) + offsets[h]
                row.append(rid)
        ids_all.append(row)
    return ids_all[-len(tokens):] if tokens else []


class DiskPleNGramEmbedding(nn.Module):
    """Drop-in replacement for ``Qwen4ExpTextNGramEmbedding``.

    The module keeps the deterministic rowid logic in Python and delegates the
    actual row fetch to EngramDB.  It maintains a minimal token history so
    sequential decode steps see the correct previous n-gram context, even when
    used outside a full Transformer ``Cache`` integration.
    """

    def __init__(
        self,
        store: Any,
        num_embeddings: int | None = None,
        embedding_dim: int = 160,
        num_heads: int = PLE_HEADS,
        layer_multipliers: list[int] | None = None,
        scale: float = 1.0,
        dtype: Any | None = None,
        cache_size: int = 4096,
        prefetch_executor: Any = None,
        prefetch_timeout: float | None = None,
        eos: int = PLE_EOS,
        prime_sizes: list[int] | None = None,
        offsets: list[int] | None = None,
        ngram_size: int = PLE_NGRAM_SIZE,
        heads_per_ngram: int = PLE_HEADS_PER_NGRAM,
        divisor: int = PLE_DIVISOR,
    ) -> None:
        super().__init__()
        if dtype is None:
            if torch is None:
                raise ImportError("DiskPleNGramEmbedding requires PyTorch")
            dtype = torch.float8_e4m3fn
        if prime_sizes is None:
            prime_sizes = head_vocab_sizes(heads=int(num_heads))
        self.prime_sizes = [int(x) for x in prime_sizes]
        self.num_heads = len(self.prime_sizes)
        self.head_offsets = (
            [int(x) for x in offsets]
            if offsets is not None
            else head_offsets(self.prime_sizes)
        )
        self.ngram_size = int(ngram_size)
        self.heads_per_ngram = int(heads_per_ngram)
        if num_embeddings is None:
            num_embeddings = padded_vocab_size(self.prime_sizes, divisor)
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.head_dim = self.embedding_dim // self.num_heads
        self.layer_multipliers = list(layer_multipliers or [23_703_573_157_769, 20_109_073_645_365, 8_052_911_324_071])
        self.scale = float(scale)
        self.eos = int(eos)
        self.divisor = int(divisor)
        self.table = DiskPleEmbedding(
            store=store,
            num_embeddings=self.num_embeddings,
            embedding_dim=self.head_dim,
            dtype=dtype,
            cache_size=cache_size,
            prefetch_executor=prefetch_executor,
            prefetch_timeout=prefetch_timeout,
        )
        self.store = store
        self._context: list[list[int]] = []
        self._last_prefetch_future: Any = None
        self._native_standard = (
            self.layer_multipliers
            == [23_703_573_157_769, 20_109_073_645_365, 8_052_911_324_071]
            and self.ngram_size == 3
            and self.heads_per_ngram == 8
            and self.eos == PLE_EOS
            and self.prime_sizes == head_vocab_sizes()
        )

    def _rows_for(self, seq: list[int], history: list[int]) -> list[list[int]]:
        """Return PLE rowids, using the native Rust path when possible."""
        if self._native_standard:
            try:
                from . import rowids_for_seq_with_history

                return rowids_for_seq_with_history(history, seq)
            except Exception:
                pass
        return ple_rowids(
            seq,
            self.layer_multipliers,
            self.eos,
            sizes=self.prime_sizes,
            offsets=self.head_offsets,
            ngram_size=self.ngram_size,
            heads_per_ngram=self.heads_per_ngram,
            history=history,
        )

    def reset_history(self) -> None:
        self._context = []

    def close(self) -> None:
        """Close the underlying disk table and stop its prefetch executor."""
        self.table.close()

    def prefetch(self, input_ids: torch.Tensor) -> Any:
        """Prefetch PLE rows for ``input_ids`` without consuming them.

        This computes the same rowids as :meth:`forward` (using the current
        per-batch n-gram context) and submits the missing disk reads to the
        background prefetcher.  It does not modify the adapter's context state.
        """
        was_1d = input_ids.dim() == 1
        if was_1d:
            input_ids = input_ids.unsqueeze(0)
        batch_size, _seq_len = input_ids.shape
        while len(self._context) < batch_size:
            self._context.append([self.eos] * (self.ngram_size - 1))
        flat_rows: list[int] = []
        for b in range(batch_size):
            seq = input_ids[b].tolist()
            rows = self._rows_for(seq, self._context[b])
            for row in rows:
                flat_rows.extend(row)
        self._last_prefetch_future = self.table.prefetch(flat_rows)
        return self._last_prefetch_future

    def forward(self, input_ids: torch.Tensor, past_key_values: Any = None) -> torch.Tensor:
        del past_key_values  # history is managed internally by this adapter
        was_1d = input_ids.dim() == 1
        if was_1d:
            input_ids = input_ids.unsqueeze(0)
        batch_size, seq_len = input_ids.shape
        while len(self._context) < batch_size:
            self._context.append([self.eos] * (self.ngram_size - 1))

        batch_rows: list[list[list[int]]] = []
        for b in range(batch_size):
            seq = input_ids[b].tolist()
            history = self._context[b]
            rows = self._rows_for(seq, history)
            batch_rows.append(rows)
            self._context[b] = (history + seq)[-(self.ngram_size - 1):]

        rids = torch.tensor(batch_rows, dtype=torch.int64)
        raw = self.table(rids).to(torch.float32)
        out = (raw * self.scale).flatten(-2)
        if was_1d:
            out = out.squeeze(0)
        return out
