"""Tests for engramdb.addressable_memory (loaded standalone).

The module is intentionally dependency-free, so the test loads it directly by
path to avoid requiring the native EngramDB extension.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_MODULE_PATH = Path(__file__).parents[1] / "engramdb" / "addressable_memory.py"
_spec = importlib.util.spec_from_file_location("addressable_memory", _MODULE_PATH)
assert _spec is not None and _spec.loader is not None
addressable_memory = importlib.util.module_from_spec(_spec)
sys.modules["addressable_memory"] = addressable_memory
_spec.loader.exec_module(addressable_memory)

AddressableNgramMemory = addressable_memory.AddressableNgramMemory


def test_continuation_distribution() -> None:
    mem = AddressableNgramMemory(min_order=2, max_order=3)
    mem.add_document([1, 2, 3, 4], value_id=0)
    dist, order = mem.continuation_distribution([1, 2, 3])
    assert order == 3
    assert abs(sum(dist.values()) - 1.0) < 1e-9


def test_longest_match_order() -> None:
    mem = AddressableNgramMemory(min_order=2, max_order=3)
    mem.add_document([1, 2, 3, 4], value_id=0)
    dist, order = mem.continuation_distribution([0, 1, 2])
    assert order == 2
    assert dist[3] == 1.0


def test_topk_and_retrieve() -> None:
    mem = AddressableNgramMemory(min_order=2, max_order=3)
    mem.add_document([1, 2, 3], value_id=7)
    assert mem.topk([1, 2], k=1) == [(3, 1.0, 2)]
    assert mem.retrieve([1, 2], top_k=1)[0].value_id == 7


def test_stats() -> None:
    mem = AddressableNgramMemory(min_order=2, max_order=2)
    mem.add_document([1, 2, 3], value_id=0)
    stats = mem.stats()
    assert stats["min_order"] == 2
    assert stats["max_order"] == 2
    assert stats["total_ngrams"] > 0
