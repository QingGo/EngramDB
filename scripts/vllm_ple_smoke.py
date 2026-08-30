import os
import torch
from transformers import Qwen3Config

from engramdb import Store
from engramdb.vllm_plugin import DiskPleEmbedding, install_vllm_ple

# 1. tiny local Qwen3 config
cfg = Qwen3Config(
    vocab_size=128,
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=1,
    num_attention_heads=2,
    num_key_value_heads=2,
    max_position_embeddings=128,
)
model_dir = "/tmp/tiny-qwen3"
os.makedirs(model_dir, exist_ok=True)
cfg.save_pretrained(model_dir)

os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29500")

# 2. create vLLM config
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

# 3. tiny EngramDB store backing the embedding table.
# For float32 hidden_size=32, one row is 32*4=128 bytes.
store_dir = "/tmp/engram-vllm-store"
os.makedirs(store_dir, exist_ok=True)
width = 128
rows = 128
with open(os.path.join(store_dir, "shard_000.bin"), "wb") as f:
    for i in range(rows):
        f.write(bytes([i % 256]) * width)
store = Store(store_dir, 1, rows, width)

# 4. real vLLM model class hook
install_vllm_ple(
    Qwen3ForCausalLM,
    store=store,
    attr_name="model.embed_tokens",
    embedding_dim=32,
    dtype=torch.float32,
)

try:
    with set_current_vllm_config(vc):
        init_distributed_environment(world_size=1, rank=0, backend="gloo")
        initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
        )
        model = Qwen3ForCausalLM(vllm_config=vc)
        emb = model.model.embed_tokens

        print("patched type:", type(emb).__name__)
        assert isinstance(emb, DiskPleEmbedding), "install_vllm_ple did not replace embed_tokens"

        out = emb(torch.tensor([[1, 2, 3]]))
        print("embed out:", tuple(out.shape), out.dtype)
        assert out.shape == (1, 3, 32)

        # 5. instance-level helper on a constructed model
        from engramdb.vllm_plugin import patch_named_embedding
        patch_named_embedding(model, "model.embed_tokens", store, embedding_dim=32, dtype=torch.float32)
        print("instance helper type:", type(model.model.embed_tokens).__name__)
        assert isinstance(model.model.embed_tokens, DiskPleEmbedding)

        print("VLLM_PLE_VERIFY_OK")
finally:
    try:
        destroy_distributed_environment()
    except Exception:
        pass
