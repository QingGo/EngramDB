#!/usr/bin/env python3
"""Faithful serving A/B: the model's OWN embedding table, served from disk.

The earlier serving A/B (probes/serve_ple_ab_session42.md) injected a **random
projection**, so its output was gibberish and no correctness statement could be
attached to it.  This one replaces ``embed_tokens`` with a disk-backed module
holding the **exact trained weights**, which were verified to round-trip bit for
bit through EngramDB (probes/ple_disk_faithfulness_session42.json).

So one run yields both halves:

  * performance   tok/s with the table in HBM vs on NVMe
  * semantics     greedy token ids must be IDENTICAL between the two arms

The second half is what makes the first half quotable.

Honest scope
------------
The input embedding is **1 row per token**, not the 16 rows/token of a real Qwen
PLE / 48 of V4.1.  This measures the cost of moving a *real, used* table to disk;
it does not reproduce PLE's I/O shape.  For that, see the synthetic arms in
serve_ple_ab.py, which are explicitly labelled as not faithful.

Usage
-----
    python serve_faithful_embed_ab.py --model /path/Qwen3.5-0.8B --iterations 3
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import torch  # noqa: E402
from torch import nn  # noqa: E402


def dump_embedding_table(model_dir: str, out_dir: str, shards: int = 2) -> dict:
    """Write the checkpoint's embedding table to an EngramDB Store."""
    from safetensors import safe_open

    import engramdb

    ckpt = sorted(Path(model_dir).glob("*.safetensors"))[0]
    with safe_open(str(ckpt), framework="pt") as f:
        key = next(k for k in f.keys() if k.endswith("embed_tokens.weight"))
        W = f.get_tensor(key)
    vocab, hidden = W.shape
    width = hidden * W.element_size()
    rps = max(1, -(-vocab // shards))
    raw = W.contiguous().view(torch.uint8).numpy().tobytes()
    for s in range(shards):
        lo, hi = s * rps, min(vocab, (s + 1) * rps)
        with open(Path(out_dir) / f"shard_{s:03d}.bin", "wb") as fh:
            fh.write(raw[lo * width : hi * width])
    store = engramdb.Store(out_dir, shards, rps, width)
    return {
        "store": store, "key": key, "vocab": vocab, "hidden": hidden,
        "dtype": W.dtype, "width": width, "shards": shards, "rows_per_shard": rps,
    }


class DiskEmbedding(nn.Module):
    """Exact drop-in for nn.Embedding, rows streamed from an EngramDB Store."""

    def __init__(self, store, vocab: int, hidden: int, dtype):
        super().__init__()
        self.store = store
        self.num_embeddings = vocab
        self.embedding_dim = hidden
        self.dtype = dtype
        self.calls = 0
        self.fetch_s = 0.0

    def forward(self, ids):
        t0 = time.perf_counter()
        flat = [int(i) for i in ids.reshape(-1)]
        buf = bytearray(self.store.fetch(flat))
        self.fetch_s += time.perf_counter() - t0
        self.calls += 1
        t = torch.frombuffer(buf, dtype=torch.uint8).view(self.dtype)
        return t.reshape(*ids.shape, self.embedding_dim)


def find_layer_list(model):
    best = (None, None)
    for name, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) >= 8:
            if best[1] is None or len(mod) > len(best[1]):
                best = (name, mod)
    return best


def find_model(llm):
    attrs = ("llm_engine", "engine_core", "engine_core_client", "model_executor",
             "driver_worker", "model_runner", "worker", "model")
    seen, q = set(), [llm]
    while q:
        o = q.pop(0)
        if id(o) in seen:
            continue
        seen.add(id(o))
        if isinstance(o, nn.Module) and find_layer_list(o)[1] is not None:
            return o
        for a in attrs:
            try:
                v = getattr(o, a, None)
            except Exception:
                continue
            if v is not None and id(v) not in seen:
                q.append(v)
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--iterations", type=int, default=3)
    ap.add_argument("--prompt-len", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--gpu-mem-util", type=float, default=0.85)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    llm = LLM(model=args.model, enforce_eager=True, dtype="bfloat16",
              gpu_memory_utilization=args.gpu_mem_util, max_model_len=4096,
              disable_log_stats=True, trust_remote_code=True)

    inner = find_model(llm)
    if inner is None:
        raise SystemExit("could not reach the model; VLLM_ENABLE_V1_MULTIPROCESSING=0 set?")
    emb = next((m for n, m in inner.named_modules() if n.endswith("embed_tokens")), None)
    if emb is None:
        raise SystemExit("no embed_tokens module found")

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompts = [{"prompt_token_ids": [1000 + (7 * i + j) % 200000 for j in range(args.prompt_len)]}
               for i in range(args.batch)]
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, ignore_eos=True)

    def run(tag):
        t0 = time.perf_counter()
        outs = llm.generate(prompts, sp)
        dt = time.perf_counter() - t0
        toks = [list(o.outputs[0].token_ids) for o in outs]
        n = sum(len(t) for t in toks)
        print(f"[run] {tag:<10} tokens={n} s={dt:.3f} tok/s={n/dt:.1f}")
        return {"tag": tag, "seconds": dt, "tokens": n, "tok_per_s": n / dt,
                "token_ids": toks}

    baseline = [run("baseline") for _ in range(args.iterations)]

    tmp = tempfile.mkdtemp(prefix="engram-faithful-")
    handle = None
    try:
        info = dump_embedding_table(args.model, tmp)
        disk = DiskEmbedding(info["store"], info["vocab"], info["hidden"], info["dtype"]).to("cuda")
        full = next(n for n, m in inner.named_modules() if m is emb)
        parent_name, _, leaf = full.rpartition(".")
        parent = inner.get_submodule(parent_name) if parent_name else inner
        setattr(parent, leaf, disk)
        print(f"[inject] {parent_name}.{leaf} -> DiskEmbedding "
              f"({info['vocab']}x{info['hidden']} {info['dtype']}, "
              f"{info['shards']} shards x {info['rows_per_shard']} rows)")
        disk_arm = [run("disk-embed") for _ in range(args.iterations)]
        setattr(parent, leaf, emb)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    b_tokens = baseline[0]["token_ids"]
    matched = all(r["token_ids"] == b_tokens for r in disk_arm)
    b_med = sorted(r["tok_per_s"] for r in baseline)[len(baseline) // 2]
    d_med = sorted(r["tok_per_s"] for r in disk_arm)[len(disk_arm) // 2]
    result = {
        "model": args.model, "engine": "vllm",
        "faithful": True,
        "enforce_eager": True,
        "rows_per_token": 1,
        "note": "input embedding only (1 row/token); NOT the 16-row PLE I/O shape",
        "table": {k: info[k] for k in ("key", "vocab", "hidden", "width", "shards")},
        "baseline_tok_per_s": b_med,
        "disk_tok_per_s": d_med,
        "delta_pct": 100.0 * (d_med - b_med) / b_med,
        "added_us_per_token": 1e6 / d_med - 1e6 / b_med,
        "greedy_token_ids_identical": matched,
        "disk_reader": {"calls": disk.calls, "fetch_s": disk.fetch_s,
                        "us_per_call": disk.fetch_s / max(1, disk.calls) * 1e6},
        "baseline_runs": [{k: v for k, v in r.items() if k != "token_ids"} for r in baseline],
        "disk_runs": [{k: v for k, v in r.items() if k != "token_ids"} for r in disk_arm],
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, indent=2) + "\n")
    return 0 if matched else 1


if __name__ == "__main__":
    raise SystemExit(main())
