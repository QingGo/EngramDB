#!/usr/bin/env python3
"""Can the n-gram hash be computed on the host, bit-identically?

Session 45 measured the whole `file-staged` exposure back to one cause: the row
ids are produced on the device, so a host read must start with a device->host
copy, which forces a sync and leaves the read with no lead time. The fix is to
produce the ids on the host instead.

That is only worth building if the host result is *bit-identical* to the
device's, so this checks exactly that, in two steps:

  A. same code path, both devices: `_hash_contexts` on CPU vs CUDA with the
     fusion kernel disabled. Integer multiply/xor/remainder should be exact on
     both, but "should" is what this file is for.
  B. fused device kernel vs the portable reference: with fusion enabled the
     device takes `fused_qwen4_ngram_hash`; compare it against step A's CPU
     answer.

EOS tokens are sprinkled through the contexts on purpose: `_hash_contexts`
routes through `_shift_right_ignore_eos`, and a host implementation has to
reproduce that window reset, not just the mixing arithmetic.

This does not test *getting the contexts* -- in the engine they come from
`ReqToTokenPool.get_ngram_context`, device-side state. The scheduler holds the
same tokens on the CPU, which is the plumbing the next patch has to do.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch


def _stub_pool(module):
    """`_hash_contexts` reaches for the global pool only to cache shift tables."""

    class _Pool:
        ple_window_cache = None

    module.get_req_to_token_pool = lambda: _Pool()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/root/autodl-tmp/qwen35-ple/models/Qwen3.5-0.8B")
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import sglang

    print(f"sglang from {os.path.dirname(sglang.__file__)}", flush=True)
    from sglang.srt.models import qwen4_exp as q
    from sglang.srt.runtime_context import publish
    from sglang.srt.server_args import ServerArgs

    publish(ServerArgs(model_path=args.model_path, tp_size=1), role="test")
    # tp_size comes from the published config; the *group* still has to exist.
    import sglang.srt.distributed as dist

    dist.init_distributed_environment(
        world_size=1,
        rank=0,
        local_rank=0,
        distributed_init_method="tcp://127.0.0.1:29517",
        backend="nccl",
    )
    dist.initialize_model_parallel(tensor_model_parallel_size=1)
    _stub_pool(q)

    os.environ["SGLANG_ENABLE_QWEN4_PLE_FUSION"] = "0"
    config = q.Qwen4ExpTextConfig(
        vocab_size=4096,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        intermediate_size=512,
        hc_count=4,
        ple_layer_ids=[2],
        ple_embed_dim=2560,
        ngram_size=3,
        heads_per_ngram=8,
        ngram_vocab_size_base=4096,
        make_ngram_vocab_size_divisible_by=128,
        eos_token_id=2,
        ple_embedding_dtype="float8_e4m3fn",
    )
    emb = q.Qwen4ExpNGramEmbedding(config, config.ple_embed_dim, ple_layer_index=0)
    print(
        f"ngram_size={emb.ngram_size} heads={emb.ngram_heads} eos={emb.eos_token_id} "
        f"fused_at_init={emb.enable_ple_fusion}",
        flush=True,
    )

    g = torch.Generator().manual_seed(args.seed)
    # 1 in 8 slots is EOS so the window-reset path is exercised.
    contexts = torch.randint(0, config.vocab_size, (args.tokens, emb.ngram_size), generator=g)
    eos_mask = torch.rand((args.tokens, emb.ngram_size), generator=g) < 0.125
    contexts[eos_mask] = emb.eos_token_id
    # A second shape: decode-sized (one row per token, one column per ngram slot).
    decode_ctx = contexts.clone()

    report = {"ngram_size": emb.ngram_size, "heads": emb.ngram_heads, "tokens": args.tokens}

    # --- A. same code path, CPU vs CUDA -------------------------------------
    # Built outside a model, the hash buffers (layer_multipliers, the per-head
    # vocab sizes and offsets) are CPU tensors; move the module, not the input,
    # so both runs use identical values.
    emb.enable_ple_fusion = False  # force the portable path on both devices
    cpu_ids = emb._hash_contexts(decode_ctx, decode_sized=True)
    emb.cuda()
    cuda_ids = emb._hash_contexts(decode_ctx.to("cuda"), decode_sized=True).cpu()
    report["A_cpu_vs_cuda_equal"] = bool(torch.equal(cpu_ids, cuda_ids))
    if not report["A_cpu_vs_cuda_equal"]:
        diff = (cpu_ids != cuda_ids)
        report["A_mismatch_count"] = int(diff.sum())
        report["A_mismatch_sample"] = [
            [int(v) for v in row] for row in torch.nonzero(diff)[:5].tolist()
        ]
    report["A_id_min"] = int(cpu_ids.min())
    report["A_id_max"] = int(cpu_ids.max())

    # A second run must be stable (no hidden state in the hash).
    emb.cpu()
    again = emb._hash_contexts(decode_ctx, decode_sized=True)
    report["A_repeatable"] = bool(torch.equal(cpu_ids, again))
    emb.cuda()

    # --- B. fused kernel vs the portable reference --------------------------
    try:
        from sglang.kernels.ops.qwen4_ple import (
            can_fuse_qwen4_ngram_hash,
            fused_qwen4_ngram_hash,
        )

        ok = can_fuse_qwen4_ngram_hash(
            decode_ctx.to("cuda"),
            emb.layer_multipliers,
            emb.ngram_heads_vocab_sizes,
            emb.ngram_heads_offsets,
        )
        report["B_can_fuse"] = bool(ok)
        if ok:
            fused = fused_qwen4_ngram_hash(
                decode_ctx.to("cuda"),
                emb.layer_multipliers,
                emb.ngram_heads_vocab_sizes,
                emb.ngram_heads_offsets,
                emb.eos_token_id,
            ).cpu()
            report["B_fused_vs_reference_equal"] = bool(torch.equal(fused, cpu_ids))
            if not report["B_fused_vs_reference_equal"]:
                report["B_mismatch_count"] = int((fused != cpu_ids).sum())
            # and through the module's own dispatch with fusion enabled
            emb.enable_ple_fusion = True
            dispatched = emb._hash_contexts(decode_ctx.to("cuda"), decode_sized=True).cpu()
            report["B_module_fused_equal"] = bool(torch.equal(dispatched, cpu_ids))
    except Exception as exc:
        report["B_error"] = repr(exc)

    print(json.dumps(report, indent=2), flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
    return 0 if report.get("A_cpu_vs_cuda_equal") else 1


if __name__ == "__main__":
    sys.exit(main())
