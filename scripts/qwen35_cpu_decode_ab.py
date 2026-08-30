#!/usr/bin/env python3
"""CPU decode A/B on a real Qwen3.5-0.8B model.

This benchmarks a real 0.8B model with its input embedding replaced by
EngramDB's disk-backed embedding.  The store is created as a sparse file so we
do not need to materialize 1GB of embedding rows; performance is measured on
the disk read path, not on exact output equivalence.

Example (inside a Torch/Transformers venv with Qwen3.5 support):

    python scripts/qwen35_cpu_decode_ab.py \
        --model /path/to/Qwen3.5-0.8B \
        --store /tmp/qwen35-store \
        --new-tokens 8
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

import engramdb
from engramdb.vllm_plugin import patch_named_embedding


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


def bench(
    model: AutoModelForCausalLM,
    seq: list[int],
    new_tokens: int,
    warmup: int = 1,
    reps: int = 3,
) -> tuple[float, int]:
    input_ids = torch.tensor([seq])
    with torch.no_grad():
        model.generate(input_ids, max_new_tokens=2, do_sample=False, use_cache=True)
    best = float("inf")
    times: list[float] = []
    for _ in range(reps):
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(
                input_ids,
                max_new_tokens=new_tokens,
                do_sample=False,
                use_cache=True,
            )
        dt = time.time() - t0
        times.append(dt)
        best = min(best, dt)
    tok = out.shape[-1] - input_ids.shape[-1]
    return tok / best, tok


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--store", required=True)
    ap.add_argument("--seq", default="1,2,3")
    ap.add_argument("--new-tokens", type=int, default=8)
    ap.add_argument("--cache-size", type=int, default=4096)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--csv", default="", help="optional CSV output path")
    args = ap.parse_args()

    seq = [int(x) for x in args.seq.split(",")]
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        low_cpu_mem_usage=True,
        dtype=torch.float32,
    )
    hidden = model.config.hidden_size
    vocab = model.get_input_embeddings().num_embeddings
    width = hidden * 4
    create_sparse_store(args.store, vocab, width)

    results: list[tuple[str, float, int]] = []
    tok_s, tok = bench(model, seq, args.new_tokens, reps=args.reps)
    print(f"memory: new_tokens={tok} tok/s={tok_s:.2f}")
    results.append(("memory", tok_s, tok))
    del model

    for label, cache_size in (("raw", 0), ("lru", args.cache_size)):
        disk_model, store = load_disk_model(
            args.model, args.store, hidden, vocab, cache_size
        )
        try:
            tok_s_disk, tok_disk = bench(
                disk_model, seq, args.new_tokens, reps=args.reps
            )
            print(
                f"disk({label},cache={cache_size}): "
                f"new_tokens={tok_disk} tok/s={tok_s_disk:.2f}"
            )
            results.append((f"disk-{label}-cache{cache_size}", tok_s_disk, tok_disk))
        finally:
            store.close()
        del disk_model

    if args.csv:
        with open(args.csv, "w", encoding="utf-8") as f:
            f.write("label,tok_s,new_tokens\n")
            for label, tok_s, tok in results:
                f.write(f"{label},{tok_s:.6f},{tok}\n")
        print(f"csv written: {args.csv}")


if __name__ == "__main__":
    main()
