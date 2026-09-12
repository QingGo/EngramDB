#!/usr/bin/env python3
"""Unit tests for real_perf_gate._evaluate (no real table required).

The gate's whole purpose is to fail when something is wrong, so its decision
logic needs its own tests: a gate that silently passes on missing data is worse
than no gate.  Run directly or via scripts/gate.sh.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from real_perf_gate import _evaluate  # noqa: E402

PASSING = {
    "rowids": "random",
    "warmup": 2,
    "reps": 5,
    "store_fetch_tokens_per_s": 92_000.0,
    "ple_memory_tokens_per_s": 76_000.0,
    "ple_memory_adapter": {
        "tokens_per_s": 48_000.0,
        "microseconds_per_token": 20.8,
    },
}

FAILURES: list[str] = []


def check(name: str, data: dict, expect_ok: bool, expect_substr: str | None = None) -> None:
    ok, failures, _ = _evaluate(data)
    if ok != expect_ok:
        FAILURES.append(f"{name}: expected ok={expect_ok}, got ok={ok} ({failures})")
        return
    if expect_substr and not any(expect_substr in f for f in failures):
        FAILURES.append(f"{name}: failures {failures} do not mention {expect_substr!r}")
        return
    print(f"  {name:52s} ok={ok!s:5s} {'| ' + '; '.join(failures) if failures else ''}")


def mutate(**kw) -> dict:
    d = {k: (dict(v) if isinstance(v, dict) else v) for k, v in PASSING.items()}
    d.update(kw)
    return d


print("=== real_perf_gate._evaluate ===")
check("healthy run passes", PASSING, True)
check("adapter key missing entirely", mutate(ple_memory_adapter={}), False, "missing")
check("adapter measured nothing (None)", mutate(ple_memory_adapter={"tokens_per_s": None}), False, "missing")
check("adapter raised", mutate(ple_memory_adapter={"error": "ImportError"}), False, "raised")
check("adapter below 20k tok/s", mutate(ple_memory_adapter={"tokens_per_s": 19_000.0}), False, "adapter")
check(
    "adapter over 500 us/token only",
    mutate(ple_memory_adapter={"tokens_per_s": 25_000.0, "microseconds_per_token": 700.0}),
    False,
    "us/token",
)
check("store_fetch too slow", mutate(store_fetch_tokens_per_s=100.0), False, "store_fetch")
check("ple_memory too slow", mutate(ple_memory_tokens_per_s=100.0), False, "ple_memory")

if FAILURES:
    print("\nFAILED:")
    for f in FAILURES:
        print(f"  - {f}")
    raise SystemExit(1)
print("\nall real_perf_gate checks passed")
