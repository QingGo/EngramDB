#!/usr/bin/env python3
"""CPU tiny-model PLE decode A/B (memory vs DiskPleEmbedding).

This is a reproducible decode benchmark on a small Qwen3 model.  It patches the
model's input embedding with EngramDB's disk-backed embedding and measures
``model.generate`` throughput.  It is not a vLLM/SGLang serving benchmark yet,
but it exercises the full prefill+decode path with the disk PLE data plane.

Baseline protocol:

* fixed random seed
* model in evaluation mode
* one short warmup generation
* at least 5 timed repetitions
* median and p90 as primary summary statistics
* optional CSV output for ``probes/cpu_tiny_baseline.csv``
* optional regression thresholds comparing disk variants to in-memory baseline

Example (inside a Torch/Transformers venv):

    python scripts/cpu_tiny_decode_ab.py \
        --model /tmp/tiny-qwen3-ab \
        --store /tmp/engram-vllm-ab \
        --new-tokens 64 \
        --reps 7 \
        --csv probes/cpu_tiny_baseline.csv
"""

from __future__ import annotations

import argparse
import os
import random
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM

import engramdb
from engramdb.vllm_plugin import patch_named_embedding


def set_seed(seed: int) -> None:
    random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def write_store(model: AutoModelForCausalLM, store_dir: str) -> None:
    """Write the model's input embedding rows to a flat EngramDB store."""
    Path(store_dir).mkdir(parents=True, exist_ok=True)
    weight = model.get_input_embeddings().weight.detach().cpu()
    with open(os.path.join(store_dir, "shard_000.bin"), "wb") as f:
        for row in weight:
            f.write(row.to(torch.float32).numpy().tobytes())


def percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        raise ValueError("empty list for percentile")
    if len(sorted_values) == 1:
        return sorted_values[0]
    k = (len(sorted_values) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = k - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def summarize_times(times: list[float], new_tokens: int) -> dict[str, float]:
    ordered = sorted(times)
    med_s = statistics.median(times)
    p90_s = percentile(ordered, 0.90)
    mean_s = statistics.fmean(times)
    return {
        "median_s": med_s,
        "p90_s": p90_s,
        "mean_s": mean_s,
        "min_s": ordered[0],
        "max_s": ordered[-1],
        "median_tok_s": new_tokens / med_s,
        "p90_tok_s": new_tokens / p90_s,
        "mean_tok_s": new_tokens / mean_s,
        "best_tok_s": new_tokens / ordered[0],
        "worst_tok_s": new_tokens / ordered[-1],
    }


def bench(
    model: AutoModelForCausalLM,
    seq: list[int],
    new_tokens: int,
    warmup: int = 1,
    reps: int = 7,
) -> tuple[dict[str, float], int]:
    model.eval()
    input_ids = torch.tensor([seq])
    with torch.no_grad():
        model.generate(input_ids, max_new_tokens=warmup, do_sample=False, use_cache=True)

    times: list[float] = []
    tok = 0
    with torch.no_grad():
        for _ in range(reps):
            t0 = time.time()
            out = model.generate(
                input_ids,
                max_new_tokens=new_tokens,
                do_sample=False,
                use_cache=True,
            )
            dt = time.time() - t0
            times.append(dt)
            tok = out.shape[-1] - input_ids.shape[-1]
    return summarize_times(times, tok), tok


def load_disk_model(
    model_dir: str,
    store_dir: str,
    hidden_size: int,
    cache_size: int,
) -> tuple[AutoModelForCausalLM, engramdb.Store]:
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    width = hidden_size * 4  # float32 bytes per row
    store = engramdb.Store(
        store_dir,
        1,
        model.get_input_embeddings().num_embeddings,
        width,
    )
    patch_named_embedding(
        model,
        "model.embed_tokens",
        store,
        embedding_dim=hidden_size,
        dtype=torch.float32,
        cache_size=cache_size,
    )
    return model, store


def slowdown(memory_tok_s: float, disk_tok_s: float) -> float:
    if disk_tok_s <= 0:
        return float("inf")
    return max(0.0, memory_tok_s / disk_tok_s - 1.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--store", required=True)
    ap.add_argument("--seq", default="5,6,7,8,9,10")
    ap.add_argument("--new-tokens", type=int, default=64)
    ap.add_argument("--cache-size", type=int, default=4096)
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--csv", default="probes/cpu_tiny_baseline.csv",
                    help="optional CSV output path; pass empty string to disable")
    ap.add_argument("--max-raw-slowdown", type=float, default=None)
    ap.add_argument("--max-lru-slowdown", type=float, default=None)
    args = ap.parse_args()

    if args.reps < 5:
        raise SystemExit(f"--reps must be >= 5 for a trustworthy baseline, got {args.reps}")
    if args.new_tokens < 1:
        raise SystemExit("--new-tokens must be >= 1")

    set_seed(args.seed)
    seq = [int(x) for x in args.seq.split(",")]
    if not seq:
        raise SystemExit("--seq must contain at least one token id")

    print(f"seed={args.seed} seq={seq} new_tokens={args.new_tokens} reps={args.reps}")

    model = AutoModelForCausalLM.from_pretrained(args.model)
    hidden = model.config.hidden_size
    write_store(model, args.store)

    results: list[dict[str, Any]] = []
    mem_stats, tok = bench(model, seq, args.new_tokens, warmup=args.warmup, reps=args.reps)
    print(f"memory: new_tokens={tok} median={mem_stats['median_s']:.4f}s "
          f"median_tok/s={mem_stats['median_tok_s']:.2f} p90_tok/s={mem_stats['p90_tok_s']:.2f}")
    results.append({
        "label": "memory",
        "cache_size": 0,
        "seed": args.seed,
        "reps": args.reps,
        "new_tokens": tok,
        **mem_stats,
    })

    for label, cache_size in (("raw", 0), ("lru", args.cache_size)):
        disk_model, store = load_disk_model(args.model, args.store, hidden, cache_size)
        try:
            disk_stats, disk_tok = bench(
                disk_model, seq, args.new_tokens, warmup=args.warmup, reps=args.reps
            )
            print(f"disk({label},cache={cache_size}): new_tokens={disk_tok} "
                  f"median={disk_stats['median_s']:.4f}s "
                  f"median_tok/s={disk_stats['median_tok_s']:.2f} "
                  f"p90_tok/s={disk_stats['p90_tok_s']:.2f}")
            results.append({
                "label": f"disk-{label}-cache{cache_size}",
                "cache_size": cache_size,
                "seed": args.seed,
                "reps": args.reps,
                "new_tokens": disk_tok,
                **disk_stats,
            })
        finally:
            store.close()

    by_label = {r["label"]: r for r in results}
    mem = by_label["memory"]
    raw_label = "disk-raw-cache0"
    lru_label = f"disk-lru-cache{args.cache_size}"
    if raw_label in by_label:
        s = slowdown(mem["median_tok_s"], by_label[raw_label]["median_tok_s"])
        mem["raw_slowdown"] = s
        by_label[raw_label]["raw_slowdown"] = s
    if lru_label in by_label:
        s = slowdown(mem["median_tok_s"], by_label[lru_label]["median_tok_s"])
        mem["lru_slowdown"] = s
        by_label[lru_label]["lru_slowdown"] = s

    if args.csv:
        fields = [
            "label", "cache_size", "seed", "reps", "new_tokens",
            "median_s", "p90_s", "mean_s", "min_s", "max_s",
            "median_tok_s", "p90_tok_s", "mean_tok_s", "best_tok_s", "worst_tok_s",
            "raw_slowdown", "lru_slowdown",
        ]
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", encoding="utf-8") as f:
            f.write(",".join(fields) + "\n")
            for r in results:
                row = []
                for name in fields:
                    val = r.get(name, "")
                    if isinstance(val, float):
                        row.append(f"{val:.6f}")
                    else:
                        row.append(str(val))
                f.write(",".join(row) + "\n")
        print(f"csv written: {args.csv}")

    ok = True
    if raw_label in by_label and args.max_raw_slowdown is not None:
        s = slowdown(mem["median_tok_s"], by_label[raw_label]["median_tok_s"])
        print(f"raw slowdown vs memory: {s:.1%} (limit {args.max_raw_slowdown:.1%})")
        if s > args.max_raw_slowdown:
            ok = False
    if lru_label in by_label and args.max_lru_slowdown is not None:
        s = slowdown(mem["median_tok_s"], by_label[lru_label]["median_tok_s"])
        print(f"lru slowdown vs memory: {s:.1%} (limit {args.max_lru_slowdown:.1%})")
        if s > args.max_lru_slowdown:
            ok = False
    if not ok:
        raise SystemExit("GATE FAIL: disk path exceeded slowdown threshold")
    if args.max_raw_slowdown is not None or args.max_lru_slowdown is not None:
        print("GATE PASS")
    else:
        print("no gate thresholds requested")


if __name__ == "__main__":
    main()
