#!/usr/bin/env python3
"""Bit-exact check for Qwen3.5-0.8B with an EngramDB-backed embedding.

This script:
1. Loads the real Qwen3.5-0.8B model.
2. Optionally fills the EngramDB store from the real ``embed_tokens`` weights.
3. Runs a short deterministic generation with the in-memory embedding.
4. Replaces ``model.embed_tokens`` with DiskPleEmbedding and runs the same
   generation.
5. Checks that the generated token sequence is identical and that the direct
   embedding output is bitwise/float-exact.

Example (WSL / transformers 5.x):

    python scripts/qwen35_bit_exact.py \
        --model /mnt/c/Users/minam/engramdb-transfer/Qwen3.5-0.8B \
        --store /tmp/qwen35-real-store \
        --fill \
        --new-tokens 8
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

import engramdb
from engramdb.vllm_plugin import patch_named_embedding


def fill_store(model: AutoModelForCausalLM, store_dir: str) -> str:
    """Write the real input embedding weights into a raw EngramDB store."""
    Path(store_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(store_dir, "shard_000.bin")
    weight = model.get_input_embeddings().weight.detach().cpu().float()
    arr = weight.numpy()
    with open(path, "wb") as f:
        arr.tofile(f)
    return path


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)


def run_generation(model: AutoModelForCausalLM, seq: list[int], new_tokens: int) -> list[int]:
    input_ids = torch.tensor([seq])
    model.eval()
    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=new_tokens,
            do_sample=False,
            use_cache=True,
        )
    return out[0].tolist()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--store", required=True)
    ap.add_argument("--seq", default="1,2,3,4,5")
    ap.add_argument("--new-tokens", type=int, default=8)
    ap.add_argument("--cache-size", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fill", action="store_true", help="fill store from real weights before checking")
    ap.add_argument("--check-rows", type=int, default=256, help="number of direct embedding rows to compare")
    args = ap.parse_args()

    set_seed(args.seed)
    seq = [int(x) for x in args.seq.split(",")]
    print(f"seed={args.seed} seq={seq} new_tokens={args.new_tokens}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        low_cpu_mem_usage=True,
        dtype=torch.float32,
    )
    hidden = model.config.hidden_size
    vocab = model.get_input_embeddings().num_embeddings
    width = hidden * 4

    # Reference memory generation before touching the store.
    ref_tokens = run_generation(model, seq, args.new_tokens)
    print(f"memory generation: {ref_tokens}")

    if args.fill:
        fill_store(model, args.store)
        print(f"store filled from real weights: {args.store}")

    orig_emb = model.get_input_embeddings()
    orig_weight = orig_emb.weight.detach().cpu()

    store = engramdb.Store(args.store, 1, vocab, width)
    patch_named_embedding(
        model,
        "model.embed_tokens",
        store,
        embedding_dim=hidden,
        dtype=torch.float32,
        cache_size=args.cache_size,
    )

    # Direct embedding comparison on a deterministic sample.
    g = torch.Generator().manual_seed(args.seed)
    rows = torch.randint(0, vocab, (args.check_rows,), generator=g).tolist()
    indices = torch.tensor(rows).reshape(1, -1)
    expected = orig_weight[rows].to(torch.float32)
    actual = model.get_input_embeddings()(indices).detach().cpu().float().reshape(-1, hidden)
    max_abs = (expected - actual).abs().max().item()
    allclose = torch.allclose(expected, actual, atol=0.0, rtol=0.0)
    print(f"direct embedding: rows={args.check_rows} max_abs={max_abs:.6e} allclose={allclose}")

    # Same generation after replacing the embedding with the disk-backed one.
    disk_tokens = run_generation(model, seq, args.new_tokens)
    print(f"disk generation:   {disk_tokens}")
    exact = list(ref_tokens) == list(disk_tokens)
    print(f"generation exact: {exact}")

    store.close()

    ok = allclose and exact and max_abs == 0.0
    if ok:
        print("BIT_EXACT_PASS")
    else:
        raise SystemExit("BIT_EXACT_FAIL")


if __name__ == "__main__":
    main()
