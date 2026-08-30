#!/usr/bin/env python3
"""Bit-exact check for the real Qwen4Exp PLE layer with EngramDB disk rows.

This script does not load the full 50GB+ model.  It:

1. Loads the PLE layer's small projection/conv weights from safetensors.
2. Loads the real PLE n-gram embedding rows through EngramDB Store.
3. Reimplements the PLE layer forward (same math as transformers Qwen4Exp).
4. Compares two paths:
   - "reference": direct raw-row reads from the extracted PLE shard files
   - "disk": EngramDB Store-backed DiskPleEmbedding
5. Checks that the final PLE layer output is bit-exact.

Run from the repository root:

    PYTHONPATH=python python3 scripts/ple_layer_bit_exact.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ref_ple_hash import head_vocab_sizes  # noqa: E402

import engramdb  # noqa: E402
from engramdb.vllm_plugin import DiskPleEmbedding  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "docs" / "real-weights-spec.json"
ROWS_DIR = ROOT / "data" / "real-rows"
MODEL_DIR = Path("/Volumes/My Passport/qwen38-ple")
LAYER = 1
PLE_LAYER_INDEX = 0


def load_spec() -> dict:
    return json.loads(SPEC.read_text())


def tensor_from_safetensors(fname: str, key: str) -> torch.Tensor:
    from safetensors import safe_open
    path = MODEL_DIR / fname
    with safe_open(str(path), framework="pt") as f:
        return f.get_tensor(key).to(torch.float32)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, group_size: int, eps: float = 1e-6) -> torch.Tensor:
    if group_size is not None:
        x = x.reshape(*x.shape[:-1], -1, group_size)
    out = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    out = out.flatten(-2) if group_size is not None else out
    out = out * (1.0 + weight.float())
    return out.to(x.dtype)


def short_conv(x: torch.Tensor, conv_w: torch.Tensor, kernel: int, dilation: int) -> torch.Tensor:
    seq_len = x.shape[1]
    x = x.transpose(1, 2)
    pad_len = (kernel - 1) * dilation
    x = F.pad(x, (pad_len, 0))
    x = x[..., -(pad_len + seq_len):]
    x = F.silu(F.conv1d(x, conv_w, dilation=dilation, groups=conv_w.shape[0]))
    return x.transpose(1, 2)


def read_row_fp32(rowid: int, scale: float) -> torch.Tensor:
    rows_per_shard = 2_500_012
    width = 160
    shard = rowid // rows_per_shard
    offset = (rowid % rows_per_shard) * width
    with open(ROWS_DIR / f"shard_{shard:03d}.bin", "rb") as f:
        f.seek(offset)
        raw = f.read(width)
    data = torch.frombuffer(bytearray(raw), dtype=torch.float8_e4m3fn).to(torch.float32)
    return data * scale


def _shift_right_ignore_eos(hist: list[int], shift: int, eos: int) -> list[int]:
    if shift == 0:
        return hist
    n = len(hist)
    prev_incl = [-1] * n
    last = -1
    for i, x in enumerate(hist):
        if x == eos:
            last = i
        prev_incl[i] = last
    out = []
    for i in range(n):
        seg_start = 0 if i == 0 else prev_incl[i - 1] + 1
        pos_in_seg = i - seg_start
        src = i - shift
        valid = pos_in_seg >= shift and src >= 0
        out.append(hist[src] if valid else eos)
    return out


def ple_rowids_py(tokens: list[int], multipliers: list[int], eos: int = 248044) -> list[list[int]]:
    ngram_size = 3
    heads_per = 8
    sizes = head_vocab_sizes()
    offsets = [0]
    for sz in sizes[:-1]:
        offsets.append(offsets[-1] + sz)
    hist = [eos, eos] + list(tokens)
    shifted = [_shift_right_ignore_eos(hist, sh, eos) for sh in range(ngram_size)]
    ids_all = []
    for pos in range(len(hist)):
        row = []
        for ngram in range(2, ngram_size + 1):
            start = (ngram - 2) * heads_per
            end = start + heads_per
            mixed = shifted[0][pos] * multipliers[0]
            for order in range(1, ngram):
                mixed ^= shifted[order][pos] * multipliers[order]
            for h in range(start, end):
                rid = (mixed % sizes[h]) + offsets[h]
                row.append(rid)
        ids_all.append(row)
    return ids_all[ngram_size - 1:]


def reference_embed(rowids: torch.Tensor, scale: float) -> torch.Tensor:
    rows = [read_row_fp32(int(r), scale) for r in rowids.reshape(-1)]
    return torch.stack(rows, dim=0).reshape(*rowids.shape, -1)


def disk_embed(rowids: torch.Tensor, disk: DiskPleEmbedding, scale: float) -> torch.Tensor:
    out = disk(rowids.to(torch.int64))
    return out.to(torch.float32) * scale


def run_ple_layer(
    input_ids: torch.Tensor,
    hidden_states: torch.Tensor,
    embedding_fn,
    weights: dict[str, torch.Tensor],
    spec: dict,
) -> torch.Tensor:
    hidden_size = spec["hidden_size"]
    hc_count = spec["hc_count"]
    ple_embed_dim = spec["ple_embed_dim"]
    eps = 1e-6

    rowids = torch.tensor(
        ple_rowids_py(input_ids[0].tolist(), spec["layer_multipliers_i64"]),
        dtype=torch.int64,
    ).unsqueeze(0)
    embeddings = embedding_fn(rowids).flatten(-2)  # [B,T,ple_embed_dim]

    key_proj = weights["key_proj"]
    value_proj = weights["value_proj"]
    norm_key_w = weights["norm_key"]
    norm_query_w = weights["norm_query"]
    norm_conv_w = weights["norm_conv"]
    conv_w = weights["conv1d"]

    key_normed = rms_norm(F.linear(embeddings, key_proj), norm_key_w, hidden_size, eps)
    key_normed = key_normed.unflatten(-1, (hc_count, hidden_size))
    value = F.linear(embeddings, value_proj)
    query_normed = rms_norm(hidden_states, norm_query_w, hidden_size, eps)
    query_normed = query_normed.unflatten(-1, (hc_count, hidden_size))

    gate = (key_normed * query_normed).sum(dim=-1, keepdim=True) / (hidden_size ** 0.5)
    gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
    gated_value = torch.sigmoid(gate) * value.unsqueeze(-2)
    gated_value_normed = rms_norm(gated_value.flatten(-2), norm_conv_w, hidden_size, eps)
    gated_value = gated_value.flatten(-2)

    conv_kernel = spec["ple_conv_kernel_size"]
    conv_dilation = spec["ngram_size"]
    output = gated_value + short_conv(gated_value_normed, conv_w, conv_kernel, conv_dilation)
    return output


def main() -> None:
    spec = load_spec()

    if not ROWS_DIR.exists():
        raise SystemExit(f"missing real PLE rows: {ROWS_DIR} (use scripts/extract_ple_rows.py)")

    # Real small PLE weights.
    weights = {
        "key_proj": tensor_from_safetensors(
            "model-00005-of-00131.safetensors",
            "model.language_model.layers.1.ple.key_proj.weight",
        ),
        "norm_key": tensor_from_safetensors(
            "model-00005-of-00131.safetensors",
            "model.language_model.layers.1.ple.norm_key.weight",
        ),
        "norm_query": tensor_from_safetensors(
            "model-00005-of-00131.safetensors",
            "model.language_model.layers.1.ple.norm_query.weight",
        ),
        "norm_conv": tensor_from_safetensors(
            "model-00005-of-00131.safetensors",
            "model.language_model.layers.1.ple.norm_conv.weight",
        ),
        "value_proj": tensor_from_safetensors(
            "model-00037-of-00131.safetensors",
            "model.language_model.layers.1.ple.value_proj.weight",
        ),
        "conv1d": tensor_from_safetensors(
            "model-00037-of-00131.safetensors",
            "model.language_model.layers.1.ple.conv1d.weight",
        ),
    }
    # layer_multipliers are in spec already; weight_scale dequantizes FP8 rows.
    with __import__("safetensors").safe_open(
        str(MODEL_DIR / "model-00005-of-00131.safetensors"), framework="pt"
    ) as f:
        scale = float(f.get_tensor("model.language_model.layers.1.ple.ple_embedding.ngram_embedding.weight_scale").to(torch.float32).item())

    # Open the real PLE Store and create a disk-backed n-gram embedding.
    store = engramdb.Store(
        str(ROWS_DIR),
        shards=128,
        rows_per_shard=2_500_012,
        width=160,
    )
    padded_vocab = (sum(head_vocab_sizes()) + 127) // 128 * 128
    disk = DiskPleEmbedding(
        store=store,
        num_embeddings=padded_vocab,
        embedding_dim=160,
        dtype=torch.float8_e4m3fn,
        cache_size=0,
    )

    # A short real-token sequence; EOS handling follows the reference.
    tokens = torch.tensor([[248044, 1000, 99999, 42, 12345]], dtype=torch.int64)
    hidden_states = torch.zeros(1, tokens.shape[1], spec["hidden_size"] * spec["hc_count"], dtype=torch.float32)

    ref_out = run_ple_layer(
        tokens, hidden_states,
        lambda rids: reference_embed(rids, scale),
        weights, spec,
    )
    disk_out = run_ple_layer(
        tokens, hidden_states,
        lambda rids: disk_embed(rids, disk, scale),
        weights, spec,
    )

    max_abs = (ref_out - disk_out).abs().max().item()
    allclose = torch.allclose(ref_out, disk_out, atol=0.0, rtol=0.0)
    print(f"ref_out shape={tuple(ref_out.shape)} dtype={ref_out.dtype}")
    print(f"max_abs={max_abs:.6e} allclose={allclose}")

    store.close()
    if allclose and max_abs == 0.0:
        print("PLE_LAYER_BIT_EXACT_PASS")
    else:
        raise SystemExit("PLE_LAYER_BIT_EXACT_FAIL")


if __name__ == "__main__":
    main()
