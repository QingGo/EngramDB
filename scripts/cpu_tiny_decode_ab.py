#!/usr/bin/env python3
"""CPU tiny-model PLE decode A/B (memory vs DiskPleEmbedding).

This is a first end-to-end decode benchmark on a small Qwen3 model.  It patches
the model's input embedding with EngramDB's disk-backed embedding and measures
``model.generate`` throughput.  It is not a vLLM/SGLang serving benchmark yet,
but it exercises the full prefill+decode path with the disk PLE data plane.

Example (inside a Torch/Transformers venv):

    python scripts/cpu_tiny_decode_ab.py \
        --model /tmp/tiny-qwen3-ab \
        --store /tmp/engram-vllm-ab \
        --new-tokens 64
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


def write_store(model: AutoModelForCausalLM, store_dir: str) -> None:
    """Write the model's input embedding rows to a flat EngramDB store."""
    Path(store_dir).mkdir(parents=True, exist_ok=True)
    weight = model.get_input_embeddings().weight.detach().cpu()
    with open(os.path.join(store_dir, "shard_000.bin"), "wb") as f:
        for row in weight:
            f.write(row.to(torch.float32).numpy().tobytes())


def bench(
    model: AutoModelForCausalLM,
    seq: list[int],
    new_tokens: int,
    warmup: int = 2,
    reps: int = 5,
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--store", required=True)
    ap.add_argument("--seq", default="5,6,7,8,9,10")
    ap.add_argument("--new-tokens", type=int, default=64)
    ap.add_argument("--cache-size", type=int, default=4096)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--csv", default="", help="optional CSV output path")
    args = ap.parse_args()

    seq = [int(x) for x in args.seq.split(",")]
    model = AutoModelForCausalLM.from_pretrained(args.model)
    hidden = model.config.hidden_size
    write_store(model, args.store)

    results: list[tuple[str, float, int]] = []
    tok_s, tok = bench(model, seq, args.new_tokens, reps=args.reps)
    print(f"memory: new_tokens={tok} tok/s={tok_s:.2f}")
    results.append(("memory", tok_s, tok))

    for label, cache_size in (("raw", 0), ("lru", args.cache_size)):
        disk_model, store = load_disk_model(args.model, args.store, hidden, cache_size)
        try:
            tok_s_disk, tok_disk = bench(disk_model, seq, args.new_tokens, reps=args.reps)
            print(
                f"disk({label},cache={cache_size}): "
                f"new_tokens={tok_disk} tok/s={tok_s_disk:.2f}"
            )
            results.append((f"disk-{label}-cache{cache_size}", tok_s_disk, tok_disk))
        finally:
            store.close()

    if args.csv:
        with open(args.csv, "w", encoding="utf-8") as f:
            f.write("label,tok_s,new_tokens\n")
            for label, tok_s, tok in results:
                f.write(f"{label},{tok_s:.6f},{tok}\n")
        print(f"csv written: {args.csv}")


if __name__ == "__main__":
    main()
