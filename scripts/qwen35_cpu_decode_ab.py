#!/usr/bin/env python3
"""CPU decode A/B on a real Qwen3.5-0.8B model.

This benchmarks a real 0.8B model with its input embedding replaced by
EngramDB's disk-backed embedding.  The store is created as a sparse file so we
do not need to materialize 1GB of embedding rows; performance is measured on
the disk read path, not on exact output equivalence.

The benchmark intentionally follows a reproducible baseline protocol:

* fixed random seed
* fixed input sequence
* model in evaluation mode
* one short warmup generation
* at least 5 timed repetitions
* median and p90 as the primary summary statistics
* optional CSV output for ``probes/qwen35_cpu_baseline.csv``
* optional regression thresholds comparing disk variants to in-memory baseline

Example (inside a Torch/Transformers venv with Qwen3.5 support):

    python scripts/qwen35_cpu_decode_ab.py \
        --model /path/to/Qwen3.5-0.8B \
        --store /tmp/qwen35-store \
        --new-tokens 8 \
        --reps 7 \
        --csv probes/qwen35_cpu_baseline.csv
"""

from __future__ import annotations

import argparse
import os
import platform
import random
import socket
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM

import engramdb
from engramdb.vllm_plugin import patch_named_embedding


def set_seed(seed: int) -> None:
    """Make the benchmark reproducible at the Python/Torch level."""
    random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def create_sparse_store(store_dir: str, num_embeddings: int, width: int) -> str:
    """Create a sparse store file of the correct logical size.

    The file is logically num_embeddings * width bytes but does not consume that
    much physical space until rows are written.  This is enough for throughput
    A/B on the disk path; for bit-exact output tests the rows should be filled
    from the real weights.
    """
    Path(store_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(store_dir, "shard_000.bin")
    size = num_embeddings * width
    with open(path, "wb") as f:
        f.truncate(size)
    return path


def percentile(sorted_values: list[float], p: float) -> float:
    """Linear-interpolation percentile; values must be sorted."""
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
    """Compute median / p90 / mean / min / max throughput from per-run seconds."""
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
    embedding: Any | None = None,
) -> tuple[dict[str, float], int, list[int]]:
    """Run a deterministic decode benchmark and return stats + generated tokens."""
    model.eval()
    input_ids = torch.tensor([seq])
    with torch.no_grad():
        model.generate(input_ids, max_new_tokens=warmup, do_sample=False, use_cache=True)

    if embedding is not None and hasattr(embedding, "reset_stats"):
        embedding.reset_stats()

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
    return summarize_times(times, tok), tok, out[0].tolist()


def load_disk_model(
    model_dir: str,
    store_dir: str,
    hidden_size: int,
    vocab_size: int,
    cache_size: int,
) -> tuple[AutoModelForCausalLM, engramdb.Store]:
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        low_cpu_mem_usage=True,
        dtype=torch.float32,
    )
    width = hidden_size * 4  # float32 bytes per row
    store = engramdb.Store(store_dir, 1, vocab_size, width)
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
    """Return positive when disk is slower than memory, e.g. 0.25 = 25% slower."""
    if disk_tok_s <= 0:
        return float("inf")
    return max(0.0, memory_tok_s / disk_tok_s - 1.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--store", required=True)
    ap.add_argument("--seq", default="1,2,3,4,5")
    ap.add_argument("--new-tokens", type=int, default=8)
    ap.add_argument("--cache-size", type=int, default=4096)
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--csv", default="probes/qwen35_cpu_baseline.csv",
                    help="optional CSV output path (default writes under probes/)")
    ap.add_argument("--no-create-store", action="store_true",
                    help="use an existing filled store instead of truncating a sparse placeholder")
    ap.add_argument("--shuffle", action="store_true",
                    help="shuffle memory/raw/lru run order to reduce order bias")
    ap.add_argument("--check-bit-exact", action="store_true",
                    help="compare disk generation output against the memory reference")
    ap.add_argument("--max-raw-slowdown", type=float, default=None,
                    help="fail if raw disk median is slower than memory by more than this fraction, e.g. 0.5 = 50%%")
    ap.add_argument("--max-lru-slowdown", type=float, default=None,
                    help="fail if LRU disk median is slower than memory by more than this fraction")
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

    cfg = AutoConfig.from_pretrained(args.model, local_files_only=True)
    text_cfg = getattr(cfg, "text_config", cfg)
    hidden = getattr(text_cfg, "hidden_size", None)
    vocab = getattr(text_cfg, "vocab_size", None)
    if hidden is None or vocab is None:
        raise SystemExit("could not determine hidden_size/vocab_size from config")
    width = hidden * 4
    if args.no_create_store:
        print(f"using existing store: {args.store}")
    else:
        create_sparse_store(args.store, vocab, width)

    results: list[dict[str, Any]] = []
    ref_tokens: list[int] | None = None
    variants: list[tuple[str, int | None]] = [("memory", None), ("raw", 0), ("lru", args.cache_size)]
    if args.shuffle:
        random.Random(args.seed).shuffle(variants)
        print(f"shuffled variant order: {[v[0] for v in variants]}")
    if args.check_bit_exact:
        # Keep the memory reference first so every disk variant can be compared.
        variants = [("memory", None)] + [v for v in variants if v[0] != "memory"]
        print("check-bit-exact forces memory first")

    for label, cache_size in variants:
        if label == "memory":
            model = AutoModelForCausalLM.from_pretrained(
                args.model,
                local_files_only=True,
                low_cpu_mem_usage=True,
                dtype=torch.float32,
            )
            mem_stats, tok, ref_tokens = bench(model, seq, args.new_tokens, warmup=args.warmup, reps=args.reps)
            print(f"memory: new_tokens={tok} median={mem_stats['median_s']:.4f}s "
                  f"median_tok/s={mem_stats['median_tok_s']:.2f} p90_tok/s={mem_stats['p90_tok_s']:.2f}")
            results.append({
                "label": "memory",
                "cache_size": 0,
                "seed": args.seed,
                "reps": args.reps,
                "new_tokens": tok,
                "host": socket.gethostname(),
                "platform": platform.platform(),
                "python": sys.version.split()[0],
                "torch": torch.__version__,
                "store_path": "",
                "shuffle": bool(args.shuffle),
                **mem_stats,
            })
            del model
            continue

        disk_model, store = load_disk_model(
            args.model, args.store, hidden, vocab, int(cache_size or 0)
        )
        try:
            disk_embed = disk_model.get_input_embeddings()
            disk_stats, disk_tok, disk_tokens = bench(
                disk_model, seq, args.new_tokens, warmup=args.warmup, reps=args.reps,
                embedding=disk_embed,
            )
            embed_stats = disk_embed.get_stats() if hasattr(disk_embed, "get_stats") else {}
            hit_rate = 0.0
            if embed_stats.get("hits", 0) + embed_stats.get("misses", 0) > 0:
                hit_rate = embed_stats["hits"] / (embed_stats["hits"] + embed_stats["misses"])
            fetch_ms = embed_stats.get("fetch_s", 0.0) * 1000.0
            convert_ms = embed_stats.get("convert_s", 0.0) * 1000.0
            bit_exact = None
            if args.check_bit_exact and ref_tokens is not None:
                bit_exact = list(ref_tokens) == list(disk_tokens)
                print(f"  bit_exact={bit_exact} memory_len={len(ref_tokens)} disk_len={len(disk_tokens)}")
            print(f"disk({label},cache={cache_size}): new_tokens={disk_tok} "
                  f"median={disk_stats['median_s']:.4f}s "
                  f"median_tok/s={disk_stats['median_tok_s']:.2f} "
                  f"p90_tok/s={disk_stats['p90_tok_s']:.2f} "
                  f"hit_rate={hit_rate:.1%} fetch_ms={fetch_ms:.2f} "
                  f"convert_ms={convert_ms:.2f}")
            results.append({
                "label": f"disk-{label}-cache{cache_size}",
                "cache_size": cache_size,
                "seed": args.seed,
                "reps": args.reps,
                "new_tokens": disk_tok,
                "host": socket.gethostname(),
                "platform": platform.platform(),
                "python": sys.version.split()[0],
                "torch": torch.__version__,
                "store_path": args.store,
                "shuffle": bool(args.shuffle),
                "calls": embed_stats.get("calls", 0),
                "hits": embed_stats.get("hits", 0),
                "misses": embed_stats.get("misses", 0),
                "hit_rate": hit_rate,
                "inserts": embed_stats.get("inserts", 0),
                "evictions": embed_stats.get("evictions", 0),
                "fetch_ms": fetch_ms,
                "convert_ms": convert_ms,
                "bit_exact": bit_exact if bit_exact is not None else "",
                **disk_stats,
            })
        finally:
            store.close()
        del disk_model

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
            "host", "platform", "python", "torch", "store_path", "shuffle",
            "median_s", "p90_s", "mean_s", "min_s", "max_s",
            "median_tok_s", "p90_tok_s", "mean_tok_s", "best_tok_s", "worst_tok_s",
            "calls", "hits", "misses", "hit_rate", "inserts", "evictions",
            "fetch_ms", "convert_ms", "bit_exact",
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

    # Regression gate (only enforced when limits are explicitly provided).
    ok = True
    if "disk-raw-cache0" in by_label and args.max_raw_slowdown is not None:
        s = slowdown(mem["median_tok_s"], by_label["disk-raw-cache0"]["median_tok_s"])
        print(f"raw slowdown vs memory: {s:.1%} (limit {args.max_raw_slowdown:.1%})")
        if s > args.max_raw_slowdown:
            ok = False
    if "disk-lru-cache" + str(args.cache_size) in by_label and args.max_lru_slowdown is not None:
        s = slowdown(
            mem["median_tok_s"],
            by_label["disk-lru-cache" + str(args.cache_size)]["median_tok_s"],
        )
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
