#!/usr/bin/env python3
"""P2：真实语料 + 官方分词器 → n-gram / PLE rowid 分布统计（M0 探针证据）。

输入：data/qwen38-ple-fp8/corpus/*.txt（真实文本）
输出：stdout + probes/p2_report.json
  - token 总数、唯一 n-gram（tuple 级）
  - bigram/trigram 频次分布: Zipf 拟合（rank 幂律系数 s）
  - PLE rowid（16 头）总命中/去重率；前 K% 唯一 rowid 覆盖比例
  - 每 token 的 16 行跨头重复率（同一头内重复多高）
"""
from __future__ import annotations
import json
import math
from collections import Counter
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from ref_ple_hash import ple_rowids, spec_from_weights, head_vocab_sizes, EOS


def zeta_fit(sorted_counts: list[int]) -> float:
    """简算：log-log 线性回归斜率 = -s"""
    xs = np.log(np.arange(1, len(sorted_counts) + 1))
    ys = np.log(np.maximum(np.array(sorted_counts), 1))
    s = -np.polyfit(xs, ys, 1)[0]
    return float(s)


def main() -> int:
    tok_dir = Path("data/qwen38-ple-fp8")
    corpus_dir = tok_dir / "corpus"
    from tokenizers import Tokenizer

    t = Tokenizer.from_file(str(tok_dir / "tokenizer.json"))
    books = sorted(corpus_dir.glob("*.txt"))

    tokens = []
    for p in books:
        txt = p.read_text(errors="ignore")
        tks = t.encode(txt).ids
        tokens.extend(tks)
        tokens.append(EOS)  # 段边界（Gutenberg 每本一个文档）

    tokens = np.array(tokens, dtype=np.int64)
    print(f"tokens: {len(tokens)}  docs: {len(books)}")

    # n-gram tuple 计数（含 EOS 段边界自然生效：跨文档已插 EOS）
    bigrams: Counter = Counter()
    trigrams: Counter = Counter()
    for i in range(2, len(tokens)):
        bigrams[(tokens[i - 1], tokens[i])] += 1
        trigrams[(tokens[i - 2], tokens[i - 1], tokens[i])] += 1
    bg_sorted = sorted(bigrams.values(), reverse=True)
    tg_sorted = sorted(trigrams.values(), reverse=True)
    bg_s = zeta_fit(bg_sorted)
    tg_s = zeta_fit(tg_sorted)
    print(f"unique bigram={len(bigrams)} trigram={len(trigrams)}")
    print(f"zipf s: bigram={bg_s:.3f} trigram={tg_s:.3f}")

    # 传统 n-gram 前 K% 覆盖（训练热集依据）
    def cov(sorted_cnt: list[int], frac: float) -> float:
        total = sum(sorted_cnt)
        acc = 0
        for i, c in enumerate(sorted_cnt, start=1):
            acc += c
            if acc >= total * frac:
                return i / len(sorted_cnt) * 100
        return 100.0

    print(f"top-1% types cover: bigram={cov(bg_sorted, 0.5):.1f}%" if bg_sorted else "")
    for f in (0.5, 0.8, 0.95):
        print(f"  coverage {f*100:.0f}% -> top {cov(bg_sorted, f):.1f}% types (bigram)")

    # PLE rowid 分布（官方哈希 + 真实偏移）
    mult = spec_from_weights(tok_dir)
    sizes = head_vocab_sizes()
    ids = ple_rowids(tokens, mult, sizes)  # [T,16]
    flat = ids.reshape(-1)
    rids = Counter(int(x) for x in flat)
    row_frac = len(rids) / max(1, ids.shape[0])
    print(f"PLE rowids: 16-head flat gets={len(flat)} unique={len(rids)} ({len(rids)/len(flat)*100:.1f}%)")
    print(f"unique rowids per-position (16 heads): avg={len(rids)/ids.shape[0]:.2f}")
    # 每头去重率
    for h in range(16):
        u = len(set(int(x) for x in ids[:, h]))
        print(f"  head {h:2d}: get={ids.shape[0]} unique={u} ({u/ids.shape[0]*100:.0f}%)")

    # 热集曲线：按频次选 K 行，覆盖多少查询
    counts_sorted = sorted(rids.values(), reverse=True)
    total_q = len(flat)
    tier_curve = {}
    for top in (100, 1_000, 10_000, 100_000, 1_000_000):
        acc = sum(counts_sorted[:top])
        tier_curve[str(top)] = round(acc / total_q * 100, 3)
    print("heat curve (top rows -> % of lookups):", tier_curve)

    rep = {
        "n_tokens": int(len(tokens)),
        "n_bigrams": len(bigrams),
        "n_trigrams": len(trigrams),
        "zipf_bigram": bg_s,
        "zipf_trigram": tg_s,
        "ple_unique_rows": len(rids),
        "tier_curve": tier_curve,
        "head_unique_percent": [round(len(set(int(x) for x in ids[:, h])) / ids.shape[0] * 100, 2) for h in range(16)],
    }
    out = Path("probes/p2_report.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(rep, indent=1))
    print("report:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
