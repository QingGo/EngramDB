#!/usr/bin/env python3
"""Create the tiny Qwen3 model used by the CPU decode A/B proxy benchmarks.

``scripts/cpu_tiny_decode_ab.py`` needs a small ``Qwen3ForCausalLM`` checkpoint.
Historically this was created ad hoc in ``/tmp`` and lost, which made the
Phase 2 proxy artifact unreproducible.  This script recreates it deterministically.

The weights are random: these benchmarks measure *throughput and integration*,
not output quality.

Usage:
    python scripts/make_tiny_qwen3.py --out /tmp/tiny-qwen3-ab
    python scripts/make_tiny_qwen3.py --out /tmp/tiny-qwen3-ab --vocab-size 152064
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output model directory")
    ap.add_argument("--vocab-size", type=int, default=152064)
    ap.add_argument("--hidden-size", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, Qwen3Config

    torch.manual_seed(args.seed)
    cfg = Qwen3Config(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        intermediate_size=args.hidden_size * 2,
        num_hidden_layers=args.layers,
        num_attention_heads=args.heads,
        num_key_value_heads=args.heads,
        max_position_embeddings=512,
        architectures=["Qwen3ForCausalLM"],
    )
    model = AutoModelForCausalLM.from_config(cfg)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)

    emb = model.get_input_embeddings().weight
    manifest = {
        "vocab_size": args.vocab_size,
        "hidden_size": args.hidden_size,
        "layers": args.layers,
        "seed": args.seed,
        "embedding_rows": int(emb.shape[0]),
        "embedding_row_bytes_fp32": args.hidden_size * 4,
        "embedding_table_bytes": int(emb.numel() * 4),
    }
    (out / "engramdb_tiny_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"wrote {out}  embedding={tuple(emb.shape)}  "
        f"table={manifest['embedding_table_bytes'] / 1e6:.1f} MB"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
