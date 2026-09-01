"""Pure-Python PLE/Engram rowid math.

This module intentionally has no third-party imports.  It provides the
deterministic Qwen PLE/Engram n-gram rowid generation used by both the
torch-facing adapter and the optional serving layer.
"""

from __future__ import annotations

from typing import Any

PLE_EOS = 248044
PLE_NGRAM_SIZE = 3
PLE_HEADS_PER_NGRAM = 8
PLE_HEADS = 16
PLE_BASE = 20_000_000
PLE_DIVISOR = 128
DEFAULT_MULTIPLIERS = [
    23_703_573_157_769,
    20_109_073_645_365,
    8_052_911_324_071,
]


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
    tokens: list[int] | Any,
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
