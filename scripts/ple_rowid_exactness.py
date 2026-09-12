#!/usr/bin/env python3
"""Rowid exactness: EngramDB's keygen vs the ENGINE's own PLE rowid code.

Why this test matters more than any throughput number
-----------------------------------------------------
The whole serving A/B so far measured *cost*: how long does it take to read N
rows per token.  Cost is worthless if we read the **wrong rows**.  A single
mismatched rowid silently corrupts every token, and no tok/s number would ever
show it.

The authoritative algorithm lives in the engine, not in our repo:
``vllm/models/qwen4_exp/nvidia/ple_layer.py`` -> ``Qwen4ExpNGramEmbedding``.
This script calls **that exact code** -- unmodified, via a shim object carrying
only the eight attributes ``compute_ngram_ids`` reads -- and compares every
rowid against ``engramdb._engramdb.rowids_for_seq`` (the Rust production path).

It is CPU-only and takes seconds.  There is no reason to ever guess here.

Coverage
--------
  A. fresh single request, several lengths (incl. n < ngram_size)
  B. sequences containing eos (segment boundaries)
  C. multi-request packed batch (the engine's real layout)
  D. decode step: explicit 2-token history + 1 new token
  E. spec geometry: multipliers, prime sizes, padded total

Exit code 0 only if every rowid matches.

Usage
-----
    python scripts/ple_rowid_exactness.py            # uses the tokenizer config
    python scripts/ple_rowid_exactness.py --config /path/config.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

DEFAULT_CONFIG = (
    "/root/autodl-tmp/qwen35-ple/models/Qwen3.8-Flash-Next-FP8-tokenizer/config.json"
)

PLE_QWEN_V1 = 1

# EngramDB's Rust constants, mirrored here ONLY so the report can show what the
# engine says next to what we say.  The comparison itself never uses these.
RUST_MULTIPLIERS = [23_703_573_157_769, 20_109_073_645_365, 8_052_911_324_071]


def build_engine_ref(text_cfg: dict, max_tokens: int = 65536, max_reqs: int = 64):
    """Return ``(call, geometry)`` where ``call`` runs vLLM's own rowid code.

    ``compute_ngram_ids`` is a plain method whose entire dependency on ``self``
    is eight attributes.  Feeding it a ``SimpleNamespace`` runs the real body
    without constructing a GPU module, a TP group, or a quant method.
    """
    from vllm.models.qwen4_exp.nvidia.ple_layer import Qwen4ExpNGramEmbedding as PLE

    ngram_size = int(text_cfg["ngram_size"])
    heads_per_ngram = int(text_cfg["heads_per_ngram"])
    ngram_heads = (ngram_size - 1) * heads_per_ngram
    ple_layer_ids = list(text_cfg["ple_layer_ids"])
    eos = int(text_cfg["eos_token_id"])
    unigram_vocab = int(text_cfg["vocab_size"])
    # seed is absent from the real config; vLLM does getattr(config, "seed", 1234)
    seed = int(text_cfg.get("seed", 1234))

    # ple_dense_layer_id: index of this layer among the sorted ple_layer_ids.
    # model.py: ple_dense_layer_id_map[abs_id] = idx; layer_idx + 1 == abs_id.
    dense_id = sorted(set(ple_layer_ids)).index(ple_layer_ids[0])

    multipliers = PLE._make_layer_multipliers(
        ngram_size=ngram_size,
        unigram_vocab_size=unigram_vocab,
        seed=seed,
        ple_dense_layer_id=dense_id,
    )
    sizes, offsets, total = PLE._make_vocab_layout(
        ngram_vocab_size_base=int(text_cfg["ngram_vocab_size_base"]),
        ngram_heads=ngram_heads,
        ple_dense_layer_id=dense_id,
    )
    divisor = int(text_cfg["make_ngram_vocab_size_divisible_by"])
    padded = ((total + divisor - 1) // divisor) * divisor

    shim = SimpleNamespace(
        ngram_size=ngram_size,
        heads_per_ngram=heads_per_ngram,
        ngram_heads=ngram_heads,
        eos_token_id=eos,
        positions_buffer=torch.arange(max_tokens, dtype=torch.int64),
        padded_buffer=torch.full((max_reqs, max_tokens), eos, dtype=torch.int64),
        ngram_heads_vocab_sizes=torch.tensor(sizes, dtype=torch.long),
        ngram_heads_offsets=torch.tensor(offsets, dtype=torch.long),
        layer_multipliers=torch.tensor(multipliers, dtype=torch.long),
        # staticmethods, so these are plain functions
        _shift_precompute=PLE._shift_precompute,
        _shift_apply=PLE._shift_apply,
    )

    def call(
        input_ids: list[int],
        query_start_loc: list[int],
        ngram_context: list[list[int]],
    ) -> list[list[int]]:
        ids = PLE.compute_ngram_ids(
            shim,
            torch.tensor(input_ids, dtype=torch.int64),
            torch.tensor(query_start_loc, dtype=torch.int64),
            torch.tensor(ngram_context, dtype=torch.int64),
        )
        return ids.tolist()

    geometry = {
        "ngram_size": ngram_size,
        "heads_per_ngram": heads_per_ngram,
        "ngram_heads": ngram_heads,
        "ple_dense_layer_id": dense_id,
        "seed": seed,
        "eos_token_id": eos,
        "unigram_vocab_size": unigram_vocab,
        "multipliers": multipliers,
        "head_sizes": sizes,
        "head_offsets": offsets,
        "total_vocab": total,
        "padded_vocab": padded,
        "divisor": divisor,
    }
    return call, geometry


def fresh_context(num_reqs: int, eos: int, width: int = 2) -> list[list[int]]:
    return [[eos] * width for _ in range(num_reqs)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    import engramdb._engramdb as kb

    cfg = json.loads(Path(args.config).read_text())
    text_cfg = cfg.get("text_config", cfg)
    engine_ids, geo = build_engine_ref(text_cfg)
    eos = geo["eos_token_id"]
    vocab = geo["unigram_vocab_size"]
    rng = random.Random(args.seed)

    report: dict = {"config": args.config, "geometry": geo, "cases": []}
    failures = 0

    def record(name: str, want: list[list[int]], got: list[list[int]]) -> None:
        nonlocal failures
        bad = 0
        first = None
        if len(want) != len(got):
            bad = abs(len(want) - len(got))
            first = {"reason": f"length {len(want)} != {len(got)}"}
        else:
            for i, (a, b) in enumerate(zip(want, got)):
                if a != b:
                    bad += 1
                    if first is None:
                        first = {
                            "position": i,
                            "engine": a,
                            "engramdb": b,
                            "differing_heads": [h for h in range(len(a)) if a[h] != b[h]],
                        }
        if bad:
            failures += 1
        status = "OK " if bad == 0 else "FAIL"
        print(f"  [{status}] {name:<44} n={len(want):<6} mismatched={bad}")
        if bad and first is not None:
            print(f"         first mismatch: {first}")
        report["cases"].append(
            {
                "name": name,
                "rows": len(want),
                "mismatched": bad,
                "first_mismatch": first,
                "verdict": "pass" if bad == 0 else "fail",
            }
        )

    def rand_tokens(n: int) -> list[int]:
        return [rng.randrange(vocab) for _ in range(n)]

    print("=" * 78)
    print("PLE rowid exactness: vLLM's own compute_ngram_ids  vs  EngramDB Rust")
    print("=" * 78)
    print(
        f"  geometry: heads={geo['ngram_heads']} eos={eos} seed={geo['seed']} "
        f"padded_vocab={geo['padded_vocab']:,}"
    )
    print(f"  engine multipliers: {geo['multipliers']}")
    print(f"  engramdb multipliers: {RUST_MULTIPLIERS}")
    mult_ok = geo["multipliers"] == RUST_MULTIPLIERS
    print(f"  multipliers identical: {mult_ok}")
    report["multipliers_match"] = mult_ok
    if not mult_ok:
        failures += 1

    print("\nA. fresh single request (cold start, context = [eos, eos])")
    for n in (1, 2, 3, 4, 8, 63, 64, 257):
        toks = rand_tokens(n)
        want = engine_ids(toks, [0, n], fresh_context(1, eos))
        got = kb.rowids_for_seq(toks, PLE_QWEN_V1)
        record(f"A n={n}", want, got)

    print("\nB. sequences containing eos (segment boundaries)")
    for n in (6, 16, 64):
        for n_eos in (1, 2, 3):
            toks = rand_tokens(n)
            for p in rng.sample(range(n), min(n_eos, n)):
                toks[p] = eos
            want = engine_ids(toks, [0, n], fresh_context(1, eos))
            got = kb.rowids_for_seq(toks, PLE_QWEN_V1)
            record(f"B n={n} eos×{n_eos} at {toks.count(eos)}", want, got)
    # eos at the very first / very last position
    for pos in (0,):
        toks = rand_tokens(9)
        toks[pos] = eos
        want = engine_ids(toks, [0, 9], fresh_context(1, eos))
        record("B eos at position 0", want, kb.rowids_for_seq(toks, PLE_QWEN_V1))
    toks = rand_tokens(9)
    toks[-1] = eos
    want = engine_ids(toks, [0, 9], fresh_context(1, eos))
    record("B eos at last position", want, kb.rowids_for_seq(toks, PLE_QWEN_V1))

    print("\nC. multi-request packed batch (engine's real layout)")
    for req_lens in ([3, 5], [1, 1, 1], [17, 4, 9, 2], [2, 2]):
        flat: list[int] = []
        starts = [0]
        ctx = []
        for L in req_lens:
            flat.extend(rand_tokens(L))
            starts.append(len(flat))
            ctx.append([eos, eos])
        want = engine_ids(flat, starts, ctx)
        got: list[list[int]] = []
        for i, L in enumerate(req_lens):
            got.extend(kb.rowids_for_seq(flat[starts[i] : starts[i + 1]], PLE_QWEN_V1))
        record(f"C packed {req_lens}", want, got)

    print("\nD. decode step: 2-token history + new token(s)")
    for hist_len in (2,):
        for n_new in (1, 2, 4):
            hist = rand_tokens(hist_len)
            new = rand_tokens(n_new)
            want = engine_ids(new, [0, n_new], [hist])
            got = kb.rowids_for_seq_with_history(hist, new, PLE_QWEN_V1)
            record(f"D hist={hist_len} new={n_new}", want, got)
    # history that contains eos
    hist = [eos, rand_tokens(1)[0]]
    new = rand_tokens(3)
    want = engine_ids(new, [0, 3], [hist])
    got = kb.rowids_for_seq_with_history(hist, new, PLE_QWEN_V1)
    record("D history contains eos", want, got)

    print("\nE. chunked prefill: a chunk boundary must not move a rowid")
    # A real engine splits one prompt across several forwards.  Chunk 2 starts
    # from the request's stored 2-token context, which is the last two tokens of
    # everything already processed -- LEFT-PADDED WITH EOS when the request has
    # processed fewer than two tokens (cut < ngram_size - 1).  Getting that pad
    # wrong is invisible until a prompt is chunked at a short boundary.
    #
    # Checked two ways, because they fail differently:
    #   (i)  against the engine, feeding it the same 2-wide context
    #   (ii) self-consistency: chunk1 ++ chunk2 must equal the single-shot rowids
    def ctx_for(prefix: list[int], width: int = 2) -> list[int]:
        return ([eos] * width + prefix)[-width:]

    for n, cut in ((16, 5), (32, 16), (64, 1), (64, 2), (64, 63)):
        toks = rand_tokens(n)
        hist = ctx_for(toks[:cut])
        tail = toks[cut:]
        want = engine_ids(tail, [0, len(tail)], [hist])
        got = kb.rowids_for_seq_with_history(hist, tail, PLE_QWEN_V1)
        record(f"E engine-match n={n} cut={cut} hist={hist[:2]}", want, got)

        whole = kb.rowids_for_seq(toks, PLE_QWEN_V1)
        joined = kb.rowids_for_seq(toks[:cut], PLE_QWEN_V1) + got
        record(f"E self-consistent n={n} cut={cut}", whole, joined)

    print("\nF. spec geometry vs the real table on disk")
    # The manifest says 128 shards x 2,500,012 rows.  vLLM pads the row count to
    # a multiple of `make_ngram_vocab_size_divisible_by`; the two must agree or
    # the last shard is short and every high rowid reads past the table.
    table_rows = 128 * 2_500_012
    geo_ok = geo["padded_vocab"] == table_rows
    print(f"  engine padded_vocab = {geo['padded_vocab']:,}")
    print(f"  on-disk table rows  = {table_rows:,}")
    print(f"  agree: {geo_ok}")
    print(f"  head_sizes[:3] = {geo['head_sizes'][:3]}  (expect [20000003, 20000023, 20000033])")
    first_primes_ok = geo["head_sizes"][:3] == [20_000_003, 20_000_023, 20_000_033]
    report["padded_vocab_matches_table"] = geo_ok
    report["first_primes_match"] = first_primes_ok
    if not (geo_ok and first_primes_ok):
        failures += 1

    report["failures"] = failures
    report["verdict"] = "IDENTICAL" if failures == 0 else "MISMATCH"
    print("\n" + "=" * 78)
    print(f"VERDICT: {report['verdict']}   (failing cases: {failures})")
    print("=" * 78)

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2))
        print(f"wrote {args.json_out}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
