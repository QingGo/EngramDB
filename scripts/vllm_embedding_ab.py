#!/usr/bin/env python3
"""A/B micro-benchmark: in-memory embedding vs EngramDB disk embedding.

Runs inside a real vLLM Qwen3ForCausalLM on CPU.  This is not full decode,
but it isolates the embedding-read cost with the real vLLM model class.

Usage:
  python3 scripts/vllm_embedding_ab.py
"""
import os
import time
import torch
from transformers import Qwen3Config

from engramdb import Store
from engramdb.vllm_plugin import DiskPleEmbedding, install_vllm_ple

os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29510")

cfg = Qwen3Config(
    vocab_size=128,
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=1,
    num_attention_heads=2,
    num_key_value_heads=2,
    max_position_embeddings=128,
)
model_dir = "/tmp/tiny-qwen3-ab"
os.makedirs(model_dir, exist_ok=True)
cfg.architectures = ["Qwen3ForCausalLM"]
cfg.save_pretrained(model_dir)

from vllm.config import ModelConfig, VllmConfig, DeviceConfig, set_current_vllm_config
from vllm.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
    destroy_distributed_environment,
)
from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM

mc = ModelConfig(
    model=model_dir,
    tokenizer=model_dir,
    hf_config_path=model_dir,
    skip_tokenizer_init=True,
    dtype="float32",
    max_model_len=128,
    enforce_eager=True,
)
vc = VllmConfig(model_config=mc, device_config=DeviceConfig(device="cpu"))

store_dir = "/tmp/engram-vllm-ab"
os.makedirs(store_dir, exist_ok=True)
width = 128  # 32 float32 values
rows = 128
with open(os.path.join(store_dir, "shard_000.bin"), "wb") as f:
    for i in range(rows):
        f.write(bytes([i % 256]) * width)
store = Store(store_dir, 1, rows, width)


def bench_embedding(emb, name: str, batch: int, n_iter: int = 200) -> None:
    ids = torch.randint(0, rows, (batch, 1))
    with torch.no_grad():
        for _ in range(10):
            emb(ids)
        t0 = time.perf_counter()
        for _ in range(n_iter):
            emb(ids)
        dt = time.perf_counter() - t0
    tokens = n_iter * batch
    print(f"{name}: batch={batch} calls={n_iter} tokens={tokens} "
          f"time={dt:.4f}s tok/s={tokens/dt:.1f} us_per_call={dt/n_iter*1e6:.1f}")


try:
    with set_current_vllm_config(vc):
        init_distributed_environment(world_size=1, rank=0, backend="gloo")
        initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)

        # 1. baseline in-memory model
        baseline = Qwen3ForCausalLM(vllm_config=vc)
        print("baseline embedding:", type(baseline.model.embed_tokens).__name__)

        # 2. disk-backed model using the real class hook
        install_vllm_ple(
            Qwen3ForCausalLM,
            store=store,
            attr_name="model.embed_tokens",
            embedding_dim=32,
            dtype=torch.float32,
        )
        disk = Qwen3ForCausalLM(vllm_config=vc, prefix="disk")
        print("disk embedding:", type(disk.model.embed_tokens).__name__)
        assert isinstance(disk.model.embed_tokens, DiskPleEmbedding)

        for batch in (1, 4, 16):
            bench_embedding(baseline.model.embed_tokens, "BASELINE", batch)
            bench_embedding(disk.model.embed_tokens, "DISK", batch)

        print("VLLM_EMBEDDING_AB_OK")
finally:
    try:
        destroy_distributed_environment()
    except Exception:
        pass
