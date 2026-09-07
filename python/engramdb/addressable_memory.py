"""PLE-2 addressable n-gram lexical memory.

This is the non-parametric, semantic-addressable PLE prototype.

Unlike a plain n-gram LM, an addressable memory keeps two things:

1. **continuation table**: context n-gram -> empirical next-token distribution;
2. **value index**: context n-gram -> external values (document ids, chunks,
   entity snippets, etc.).

The PLE idea is preserved exactly: the key is a sparse, discrete, exact n-gram
address; the value is the external memory content that can be retrieved without
training.  The module is intentionally dependency-free so it can be unit tested
and later plugged into RAG/serving.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class MemoryMatch:
    """A single retrieved value for an addressable n-gram key."""

    value_id: int
    order: int
    count: int
    score: float


class AddressableNgramMemory:
    """Exact n-gram addressable memory with continuation and value retrieval."""

    def __init__(self, min_order: int = 2, max_order: int = 4) -> None:
        if min_order < 1:
            raise ValueError("min_order must be >= 1")
        if max_order < min_order:
            raise ValueError("max_order must be >= min_order")
        self.min_order = int(min_order)
        self.max_order = int(max_order)
        # ngram context -> next-token counter
        self.next_counts: list[dict[tuple[int, ...], Counter[int]]] = [
            defaultdict(Counter) for _ in range(self.max_order + 1)
        ]
        # ngram context -> value id -> occurrence count
        self.value_index: list[dict[tuple[int, ...], Counter[int]]] = [
            defaultdict(Counter) for _ in range(self.max_order + 1)
        ]
        self.total_ngrams = 0

    def add_sequence(self, tokens: Iterable[int], value_id: int | None = None) -> None:
        """Add one token sequence.

        ``value_id`` is an external memory identifier (document id, chunk id,
        entity id).  If omitted, a synthetic per-sequence id is not assigned;
        value retrieval simply remains empty.
        """
        tokens = [int(t) for t in tokens]
        for i, tok in enumerate(tokens):
            if i == 0:
                continue
            for order in range(1, self.max_order + 1):
                if i < order:
                    continue
                ctx = tuple(tokens[i - order : i])
                self.next_counts[order][ctx][tok] += 1
                if value_id is not None:
                    self.value_index[order][ctx][int(value_id)] += 1
                self.total_ngrams += 1

    def add_document(self, tokens: Iterable[int], value_id: int) -> None:
        self.add_sequence(tokens, value_id=value_id)

    def _best_order(self, context: tuple[int, ...]) -> int:
        """Longest order that has a continuation count for this context."""
        n = min(self.max_order, len(context))
        while n >= self.min_order:
            if self.next_counts[n].get(context[-n:]):
                return n
            n -= 1
        return 0

    def continuation_distribution(
        self, context: Iterable[int]
    ) -> tuple[dict[int, float], int] | None:
        ctx = tuple(int(x) for x in context)
        order = self._best_order(ctx)
        if order == 0:
            return None
        c = self.next_counts[order].get(ctx[-order:])
        if not c:
            return None
        total = sum(c.values())
        return {tok: cnt / total for tok, cnt in c.items()}, order

    def topk(
        self, context: Iterable[int], k: int = 5
    ) -> list[tuple[int, float, int]]:
        """Return ``(token, prob, matched_order)`` for the top-k continuations."""
        result = self.continuation_distribution(context)
        if result is None:
            return []
        dist, order = result
        return [(tok, p, order) for tok, p in sorted(dist.items(), key=lambda x: x[1], reverse=True)[:k]]

    def retrieve(
        self,
        context: Iterable[int],
        top_k: int = 5,
        *,
        min_count: int = 1,
        order_weights: dict[int, float] | None = None,
    ) -> list[MemoryMatch]:
        """Retrieve external values whose n-gram keys match this context.

        Scores are summed over all matching orders, weighted by n-gram order
        (longer/more specific keys score higher by default).
        """
        ctx = tuple(int(x) for x in context)
        if order_weights is None:
            order_weights = {o: float(o) for o in range(self.min_order, self.max_order + 1)}
        scores: dict[int, float] = {}
        counts: dict[tuple[int, int], int] = {}
        for order in range(self.min_order, self.max_order + 1):
            if len(ctx) < order:
                continue
            key = ctx[-order:]
            counter = self.value_index[order].get(key)
            if not counter:
                continue
            w = order_weights.get(order, 1.0)
            for value_id, cnt in counter.items():
                if cnt < min_count:
                    continue
                scores[value_id] = scores.get(value_id, 0.0) + w * cnt
                counts[(value_id, order)] = cnt
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        out: list[MemoryMatch] = []
        for value_id, score in ranked:
            best_order = max(
                (o for (v, o) in counts if v == value_id and counts[(v, o)] >= min_count),
                default=0,
            )
            count = max(counts.get((value_id, o), 0) for o in range(self.min_order, self.max_order + 1))
            out.append(
                MemoryMatch(
                    value_id=value_id,
                    order=best_order,
                    count=count,
                    score=float(score),
                )
            )
        return out

    def stats(self) -> dict[str, int]:
        return {
            "min_order": self.min_order,
            "max_order": self.max_order,
            "total_ngrams": self.total_ngrams,
            "unique_continuations": sum(len(t) for t in self.next_counts),
            "unique_value_keys": sum(len(t) for t in self.value_index),
        }


__all__ = ["AddressableNgramMemory", "MemoryMatch"]
