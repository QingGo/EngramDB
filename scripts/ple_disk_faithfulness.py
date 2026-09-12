#!/usr/bin/env python3
"""Does the disk path return the model's *own* weights, bit for bit?

Phase A's first sub-condition is "use the model's real table, not a random
projection".  The serving A/B injected a random projection, so its output was
gibberish and no correctness claim could follow from it.

This is the CPU-only half of that condition: take the trained embedding table
straight out of the checkpoint, put it behind EngramDB, read it back, and
compare **bit for bit**.  No GPU, no vLLM.

Two levels, because they fail differently:

  1. byte round-trip   store row r, fetch rowid r, compare raw bytes
                       -> catches addressing / shard-mapping / width errors
  2. module swap       nn.Embedding(weight=W) vs disk-backed module
                       -> catches dtype/reshape/offset errors that a byte
                          comparison of the same code path would not

Usage
-----
    python scripts/ple_disk_faithfulness.py --model /path/to/Qwen3.5-0.8B
"""

from __future__ import annotations

import argparse
import json
import hashlib
import shutil
import tempfile
from pathlib import Path

SHARD_ROWS = 250_012  # ~250k rows per shard keeps shard count sane at 248k vocab


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--shards", type=int, default=2)
    args = ap.parse_args()

    import torch
    from safetensors import safe_open

    import engramdb

    ckpt = None
    for cand in sorted(Path(args.model).glob("*.safetensors")):
        ckpt = cand
        break
    if ckpt is None:
        raise SystemExit(f"no .safetensors under {args.model}")

    with safe_open(str(ckpt), framework="pt") as f:
        keys = [k for k in f.keys() if k.endswith("embed_tokens.weight")]
        if not keys:
            raise SystemExit(f"no embed_tokens.weight in {ckpt}; keys: {list(f.keys())[:5]}")
        key = keys[0]
        W = f.get_tensor(key)
    print(f"[model] {key} shape={tuple(W.shape)} dtype={W.dtype}")

    vocab, hidden = W.shape
    itemsize = W.element_size()
    width = hidden * itemsize
    rows_per_shard = max(1, -(-vocab // args.shards))
    raw = W.contiguous().view(torch.uint8).numpy().tobytes()
    print(
        f"[table] vocab={vocab} hidden={hidden} width={width}B "
        f"total={len(raw) / 2**20:.1f} MiB -> {args.shards} shards x {rows_per_shard} rows"
    )

    td = tempfile.mkdtemp(prefix="engram-faithful-")
    try:
        for s in range(args.shards):
            lo, hi = s * rows_per_shard, min(vocab, (s + 1) * rows_per_shard)
            with open(Path(td) / f"shard_{s:03d}.bin", "wb") as fh:
                fh.write(raw[lo * width : hi * width])
        store = engramdb.Store(td, args.shards, rows_per_shard, width)

        # ---- level 1: byte round-trip over EVERY row ---------------------- #
        got = store.fetch(list(range(vocab)))
        byte_ok = bytes(got) == raw
        print(
            f"[1/2] byte round-trip over all {vocab} rows: "
            f"{'IDENTICAL' if byte_ok else 'MISMATCH'}"
        )
        if not byte_ok:
            # locate the first differing row so the failure is actionable
            gb = bytes(got)
            bad = next(
                (r for r in range(vocab) if gb[r * width : (r + 1) * width] != raw[r * width : (r + 1) * width]),
                None,
            )
            print(f"      first differing row: {bad}")

        # ---- level 2: module swap ----------------------------------------- #
        class DiskEmbedding(torch.nn.Module):
            def __init__(self, st, vocab, hidden, dtype):
                super().__init__()
                self.store = st
                self.num_embeddings = vocab
                self.embedding_dim = hidden
                self.dtype = dtype

            def forward(self, ids):
                flat = [int(i) for i in ids.reshape(-1)]
                buf = bytearray(self.store.fetch(flat))
                t = torch.frombuffer(buf, dtype=torch.uint8).view(self.dtype)
                return t.reshape(*ids.shape, self.embedding_dim)

        mem = torch.nn.Embedding(vocab, hidden).to(W.dtype)
        with torch.no_grad():
            mem.weight.copy_(W)
        disk = DiskEmbedding(store, vocab, hidden, W.dtype)

        ids = torch.randint(0, vocab, (4, 32))
        with torch.no_grad():
            a, b = mem(ids), disk(ids)
        mod_ok = bool(torch.equal(a, b))
        print(
            f"[2/2] module swap (nn.Embedding vs disk-backed), ids={tuple(ids.shape)}: "
            f"{'IDENTICAL' if mod_ok else 'MISMATCH'}"
        )
        if not mod_ok:
            print(f"      max abs diff = {(a.float() - b.float()).abs().max().item()}")

        store.close()
        result = {
            "model": args.model,
            "weight_key": key,
            "shape": [vocab, hidden],
            "dtype": str(W.dtype),
            "row_bytes": width,
            "shards": args.shards,
            "rows_per_shard": rows_per_shard,
            "sha256_of_table": hashlib.sha256(raw).hexdigest(),
            "byte_roundtrip_identical": byte_ok,
            "module_swap_identical": mod_ok,
            "verdict": "FAITHFUL" if (byte_ok and mod_ok) else "NOT FAITHFUL",
        }
        print(json.dumps(result, indent=2))
        if args.json_out:
            Path(args.json_out).write_text(json.dumps(result, indent=2) + "\n")
        return 0 if (byte_ok and mod_ok) else 1
    finally:
        shutil.rmtree(td, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
