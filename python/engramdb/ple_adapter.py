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

import math
from typing import Any

import torch
from torch import nn

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


def head_vocab_sizes() -> list[int]:
    return [nth_prime_after(PLE_BASE - 1, i + 1) for i in range(PLE_HEADS)]


def head_offsets(sizes: list[int] | None = None) -> list[int]:
    sizes = sizes or head_vocab_sizes()
    offsets = [0]
    for s in sizes[:-1]:
        offsets.append(offsets[-1] + s)
    return offsets


def padded_vocab_size() -> int:
    sizes = head_vocab_sizes()
    total = sum(sizes)
    return (total + PLE_DIVISOR - 1) // PLE_DIVISOR * PLE_DIVISOR


def disk_ple_from_discovery(
    store: Any,
    info: dict[str, Any],
    layer_multipliers: list[int] | None = None,
    scale: float = 1.0,
    cache_size: int = 4096,
) -> "DiskPleNGramEmbedding":
    """Build a disk PLE adapter from metadata returned by ``discover_ple``.

    This is the automatic path: feed in the output of
    ``engramdb.ple_discovery.discover_ple(model_dir)`` and it derives the
    embedding width and number of n-gram heads from the real model config.
    """
    from .ple_discovery import discover_ple  # noqa: F401  (kept for discoverability)

    if not info:
        raise ValueError("PLE discovery returned no PLE metadata")
    ngram_heads = (int(info["ngram_size"]) - 1) * int(info["heads_per_ngram"])
    return DiskPleNGramEmbedding(
        store=store,
        embedding_dim=int(info["ple_embed_dim"]),
        num_heads=ngram_heads,
        layer_multipliers=layer_multipliers,
        scale=scale,
        cache_size=cache_size,
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
) -> list[list[int]]:
    """Return [T, PLE_HEADS] rowids for a token sequence (cold path)."""
    sizes = head_vocab_sizes()
    offsets = head_offsets(sizes)
    hist = [eos, eos] + list(tokens)
    shifted = [_shift_right_ignore_eos(hist, sh, eos) for sh in range(PLE_NGRAM_SIZE)]
    ids_all = []
    for pos in range(len(hist)):
        row = []
        for ngram in range(2, PLE_NGRAM_SIZE + 1):
            start = (ngram - 2) * PLE_HEADS_PER_NGRAM
            end = start + PLE_HEADS_PER_NGRAM
            mixed = shifted[0][pos] * multipliers[0]
            for order in range(1, ngram):
                mixed ^= shifted[order][pos] * multipliers[order]
            for h in range(start, end):
                rid = (mixed % sizes[h]) + offsets[h]
                row.append(rid)
        ids_all.append(row)
    return ids_all[PLE_NGRAM_SIZE - 1:]


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
        dtype: Any = torch.float8_e4m3fn,
        cache_size: int = 4096,
        eos: int = PLE_EOS,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings or padded_vocab_size()
        self.embedding_dim = int(embedding_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.embedding_dim // self.num_heads
        self.layer_multipliers = list(layer_multipliers or [23_703_573_157_769, 20_109_073_645_365, 8_052_911_324_071])
        self.scale = float(scale)
        self.eos = int(eos)
        self.ngram_size = PLE_NGRAM_SIZE
        self.heads_per_ngram = PLE_HEADS_PER_NGRAM
        self.table = DiskPleEmbedding(
            store=store,
            num_embeddings=self.num_embeddings,
            embedding_dim=self.head_dim,
            dtype=dtype,
            cache_size=cache_size,
        )
        self._context: list[int] = [self.eos] * (self.ngram_size - 1)

    def reset_history(self) -> None:
        self._context = [self.eos] * (self.ngram_size - 1)

    def forward(self, input_ids: torch.Tensor, past_key_values: Any = None) -> torch.Tensor:
        del past_key_values  # history is managed internally by this adapter
        tokens = input_ids.reshape(-1).tolist()
        token_history = self._context + tokens
        rowids = ple_rowids(tokens, self.layer_multipliers, self.eos)
        self._context = token_history[-(self.ngram_size - 1):]

        rids = torch.tensor(rowids, dtype=torch.int64).unsqueeze(0)
        raw = self.table(rids).to(torch.float32)
        return (raw * self.scale).flatten(-2)
