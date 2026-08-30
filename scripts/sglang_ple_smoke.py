import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["SGLANG_USE_CPU_ENGINE"] = "1"
import torch
from transformers import Qwen3Config

from engramdb import Store
from engramdb.sglang import install_sglang_ple
from engramdb.vllm_plugin import DiskPleEmbedding

os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29508")

from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.srt.models.qwen3 import Qwen3ForCausalLM
from sglang.srt.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
    destroy_distributed_environment,
)
from sglang.srt.layers.dp_attention import initialize_dp_attention

model_dir = "/tmp/tiny-qwen3-sg"
os.makedirs(model_dir, exist_ok=True)
cfg = Qwen3Config(
    vocab_size=128,
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=1,
    num_attention_heads=2,
    num_key_value_heads=2,
    max_position_embeddings=128,
)
cfg.architectures = ["Qwen3ForCausalLM"]
cfg.save_pretrained(model_dir)

args = ServerArgs(
    model_path=model_dir,
    tokenizer_path=model_dir,
    skip_tokenizer_init=True,
    device="cpu",
    dtype="float32",
    tp_size=1,
    pp_size=1,
    dp_size=1,
    attn_cp_size=1,
    ep_size=1,
    load_format="dummy",
)
set_global_server_args_for_scheduler(args)
model_config = args.get_model_config()

store_dir = "/tmp/engram-sglang-store"
os.makedirs(store_dir, exist_ok=True)
width = 128
rows = 128
with open(os.path.join(store_dir, "shard_000.bin"), "wb") as f:
    for i in range(rows):
        f.write(bytes([i % 256]) * width)
store = Store(store_dir, 1, rows, width)

install_sglang_ple(
    Qwen3ForCausalLM,
    store=store,
    attr_name="model.embed_tokens",
    embedding_dim=32,
    dtype=torch.float32,
)

try:
    init_distributed_environment(world_size=1, rank=0, backend="gloo")
    initialize_model_parallel(
        tensor_model_parallel_size=1,
        expert_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        backend="gloo",
    )
    initialize_dp_attention(args, model_config)

    model = Qwen3ForCausalLM(cfg)
    emb = model.model.embed_tokens

    print("patched type:", type(emb).__name__)
    assert isinstance(emb, DiskPleEmbedding), "install_sglang_ple did not replace embed_tokens"

    out = emb(torch.tensor([[1, 2, 3]]))
    print("embed out:", tuple(out.shape), out.dtype)
    assert out.shape == (1, 3, 32)

    # instance-level helper also works
    from engramdb.vllm_plugin import patch_named_embedding
    patch_named_embedding(model, "model.embed_tokens", store, embedding_dim=32, dtype=torch.float32)
    assert isinstance(model.model.embed_tokens, DiskPleEmbedding)
    print("instance helper type:", type(model.model.embed_tokens).__name__)

    print("SGLANG_PLE_VERIFY_OK")
finally:
    try:
        destroy_distributed_environment()
    except Exception:
        pass
