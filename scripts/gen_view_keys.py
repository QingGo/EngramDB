#!/usr/bin/env python3
"""Regenerate the flat Store-P keys file used by ``engramdb view build``.

The view builder uses a fixed LCG to generate three token ids per gram and then
writes only the first token's 16 rowids (matching the Rust implementation).
This script reproduces that stream so DiskSlotIndex can be built/verified
against the actual full Store-P view without storing a multi-GB keys file in git.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from engramdb.ple_math import (
    DEFAULT_MULTIPLIERS,
    head_offsets,
    head_vocab_sizes,
    ple_rowids,
)

MASK64 = (1 << 64) - 1
MULT = 6364136223846793005
INC = 1442695040888963407
SEED = 0xDEADBEEF_1234_5678
VOCAB = 248_320


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--grams", type=int, default=20_000_096)
    parser.add_argument("--start", type=int, default=0)
    args = parser.parse_args()

    primes = head_vocab_sizes()
    offsets = head_offsets(primes)
    state = SEED
    # Fast-forward to the requested start position.
    for _ in range(args.start):
        for _ in range(3):
            state = (state * MULT + INC) & MASK64

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for _ in range(args.grams):
            a = (state := (state * MULT + INC) & MASK64) % VOCAB
            b = (state := (state * MULT + INC) & MASK64) % VOCAB
            c = (state := (state * MULT + INC) & MASK64) % VOCAB
            rows = ple_rowids(
                [int(a), int(b), int(c)],
                list(DEFAULT_MULTIPLIERS),
                sizes=primes,
                offsets=offsets,
            )
            for r in rows[0]:
                f.write(f"{r}\n")
    print(f"generated {args.grams} grams -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
