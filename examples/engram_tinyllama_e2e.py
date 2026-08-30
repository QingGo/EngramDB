
#!/usr/bin/env python3
"""TinyLlama + engram-peft + EngramDB disk-backed embeddings E2E.

This is a real end-to-end smoke: load a cached TinyLlama model, inject an
Engram layer whose MultiHeadEmbedding is backed by an EngramDB Store, run a
forward pass (and optionally a few generated tokens).

Run with the prepared Python 3.12 environment:
    PYTHONPATH=.:python:<engram-peft/src> \
    python examples/engram_tinyllama_e2e.py

Verified locally with Python 3.12 + stable torch 2.2.2 (plus the small
RMSNorm fallback in this script).  It loads TinyLlama from the local HF cache,
injects an Engram layer backed by EngramDB, runs a forward pass, and generates
a short continuation.
"""

from __future__ import annotations

import os
import struct
import tempfile
import typing
from pathlib import Path

import typing_extensions

if not hasattr(typing, "override"):
    typing.override = typing_extensions.override

import torch
import torch.nn as nn

# Older stable torch (e.g. 2.2.x) may not expose nn.RMSNorm.
if not hasattr(nn, "RMSNorm"):
    class _RMSNorm(nn.Module):
        def __init__(self, dim: int, eps: float = 1e-6):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(dim))
            self.eps = eps

        def forward(self, x):
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

    nn.RMSNorm = _RMSNorm

from transformers import AutoModelForCausalLM, AutoTokenizer

import engramdb
from engram_peft import EngramConfig, get_engram_model

# Reuse the disk MultiHeadEmbedding implementation from the interop example.
from examples.interop_engram_peft import install_disk_multi_head_embedding

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"


def build_store(total_rows: int, row_width: int) -> tuple[engramdb.Store, str]:
    directory = tempfile.mkdtemp(prefix="engramdb-tinyllama-")
    Path(directory).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(directory, "shard_000.bin"), "wb") as f:
        for i in range(total_rows):
            f.write(bytes((j + i) % 251 for j in range(row_width)))
    return engramdb.Store(directory, 1, total_rows, row_width), directory


def main() -> None:
    print(f"loading {MODEL_NAME} from cache ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )

    config = EngramConfig(
        target_layers=[0],
        hidden_size=model.config.hidden_size,
        embedding_dim=256,
        ngram_sizes=[2, 3],
        n_head_per_ngram=2,
        engram_vocab_size_per_ngram=[32, 32],
        enable_tokenizer_compression=False,
        pad_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0,
    )

    total_heads = len(config.ngram_sizes) * config.n_head_per_ngram
    per_head = config.embedding_dim // total_heads
    store, _ = build_store(4096, per_head * 4)

    install_disk_multi_head_embedding(store)

    print("injecting Engram layers with disk-backed MultiHeadEmbedding ...")
    engram_model = get_engram_model(model, config, tokenizer, train_mode="engram_only")

    text = "Once upon a time"
    input_ids = tokenizer(text, return_tensors="pt").input_ids

    with torch.no_grad():
        out = engram_model(input_ids)
    print("forward OK, logits shape:", tuple(out.logits.shape))

    # Optional tiny generation to prove the full path works.
    with torch.no_grad():
        generated = engram_model.generate(
            input_ids,
            max_new_tokens=8,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    print("generated:", tokenizer.decode(generated[0], skip_special_tokens=True))

    store.close()


if __name__ == "__main__":
    main()
