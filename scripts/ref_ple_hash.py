#!/usr/bin/env python3
"""Qwen3.8-Flash-Next PLE 哈希参考实现（数值级一致，与 transformers qwen4_exp 官方代码对齐）。

P0 结论（见 docs/engram-specs.md §3）：
- 乘子：来自 checkpoint（`layer_multipliers` I64[3]），运行期以权重为准（官方 nn.Buffer 加载）
- 头素数：`nth_prime_after(base=20_000_000-1, i+1)` i=0..15（与 GGUF 元数据一致）
- offset：素数前缀和；总槽位 320,001,001 → 官方 Embedding padded(128) = 320,001,024
  checkpoint 实际保存 2,500,012×128 = 320,001,536 行（=512 网格），尾部 512 行不索引
- EOS=248044；段语义：`_shift_right_ignore_eos`，越界回填 EOS
输出：每条目 16 个 ngram_ids（含 offsets，可直接用于 Store-I 行号）
"""
from __future__ import annotations
import math
import json
from pathlib import Path
import numpy as np

EOS = 248044
NGramSize = 3
HEADS_PER = 8
HEADS = 16
BASE = 20_000_000
DIV = 128
SHARDS = 128
ROWS_PER_SHARD = 2_500_012


def _is_prime(v: int) -> bool:
    if v < 2:
        return False
    if v % 2 == 0:
        return v == 2
    for d in range(3, math.isqrt(v) + 1, 2):
        if v % d == 0:
            return False
    return True


def nth_prime_after(start: int, count: int) -> int:
    p = start
    for _ in range(count):
        p += 1
        while not _is_prime(p):
            p += 1
    return p


def head_vocab_sizes(n=HEADS) -> list[int]:
    return [nth_prime_after(BASE - 1, i + 1) for i in range(n)]


def spec_from_weights(dir_or_json):
    """从 extract_ple_spec 的输出 json 读取（multipliers 以权重为准）。"""
    if isinstance(dir_or_json, dict):
        data = dir_or_json
    else:
        p = Path(dir_or_json)
        cands = [p, p / "real-weights-spec.json", p.parent / "docs" / "real-weights-spec.json",
                 Path("docs/real-weights-spec.json")]
        js = next((c for c in cands if c.exists() and c.is_file()), None)
        if js is None:
            raise FileNotFoundError(f"找不到 spec json: {cands}")
        data = json.loads(js.read_text())
    mult = data.get("layer_multipliers_i64")
    assert mult, "需要真实 multipliers（来自 checkpoint 权重）"
    return [int(x) for x in mult]


def ple_rowids(tokens: np.ndarray, multipliers: list[int], prime_sizes: list[int] | None = None) -> np.ndarray:
    """tokens: int64 [T]；返回 rowids [T, 16]（含头偏移）。复刻官方 forward（无 cache / pref-LL cold）。"""
    sizes = prime_sizes or head_vocab_sizes()
    offsets = np.cumsum([0] + sizes[:-1]).astype(np.int64)
    mult = np.array(multipliers, dtype=np.int64)
    ctx = np.concatenate([[EOS, EOS], tokens])  # previous_context = eos, eos
    T_out = tokens.shape[0]

    shifted = []
    for shift in range(NGramSize):
        shifted.append(_shift_right_ignore_eos(ctx, shift))

    blocks = []
    for ngram in range(2, NGramSize + 1):
        s0 = (ngram - 2) * HEADS_PER
        s1 = s0 + HEADS_PER
        mixed = shifted[0] * mult[0]
        for pos in range(1, ngram):
            mixed = np.bitwise_xor(mixed, shifted[pos] * mult[pos])
        ngram_ids = np.remainder(mixed[:, None], np.array(sizes[s0:s1])[None, :]) + offsets[s0:s1][None, :]
        blocks.append(ngram_ids)
    ids = np.concatenate(blocks, axis=-1)
    return ids[-T_out:]


def _shift_right_ignore_eos(hist: np.ndarray, shift: int) -> np.ndarray:
    if shift == 0:
        return hist
    positions = np.arange(hist.shape[0], dtype=np.int64)
    eos_pos = np.where(hist == EOS, positions, -1)
    prev_incl = np.maximum.accumulate(eos_pos)
    prev = np.concatenate([[-1], prev_incl[:-1]])
    seg_start = prev + 1
    pos_in_seg = positions - seg_start
    src_pos = positions - shift
    gather_pos = np.clip(src_pos, 0, None)
    shifted = hist[gather_pos]
    valid = (pos_in_seg >= shift) & (src_pos >= 0)
    return np.where(valid, shifted, EOS)


if __name__ == "__main__":
    import sys
    _dir = sys.argv[1] if len(sys.argv) > 1 else "data/qwen38-ple-fp8"
    mult = spec_from_weights(_dir)
    sizes = head_vocab_sizes()
    print("multipliers:", mult)
    print("primes:", sizes[:5], "...")
    print("sum:", sum(sizes), "padded128:", math.ceil(sum(sizes) / DIV) * DIV)
    demo = np.array([248044, 248044, 1000, 99999, 42], dtype=np.int64)
    ids = ple_rowids(demo, mult, sizes)
    print("sample rowids (last row):", ids[-1].tolist())
    assert int(ids.max()) < 320_001_024, "rowids 必须全部落在官方 padded 空间内"
    print("OK: all rowids < 320,001,024")
