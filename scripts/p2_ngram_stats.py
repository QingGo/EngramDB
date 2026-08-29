#!/usr/bin/env python3
"""P2 v2：多域高质语料统计（fineweb/zh/agent），按域分别出 Zipf、PLE rowid 热集、16 头唯一率；
并输出汇总对比。

用法:
    python3 scripts/p2_ngram_stats.py --dirs fineweb zh agent \
        --root data/corpus-build/text --tok-cap 60_000_000
（默认 --dirs 拉取 build manifest 的目录名；每域 token 上限防内存爆）
"""
from __future__ import annotations
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from ref_ple_hash import ple_rowids, spec_from_weights, head_vocab_sizes, EOS


def zeta_fit(sorted_counts: list[int]) -> float:
    xs = np.log(np.arange(1, len(sorted_counts) + 1))
    ys = np.log(np.maximum(np.array(sorted_counts), 1))
    try:
        s = -np.polyfit(xs, ys, 1)[0]
    except Exception:
        s = float("nan")
    return float(s)


def cov_percent(sorted_cnt: list[int], frac: float) -> float:
    total = sum(sorted_cnt)
    if total == 0:
        return 0.0
    acc = 0
    for i, c in enumerate(sorted_cnt, start=1):
        acc += c
        if acc >= total * frac:
            return i / len(sorted_cnt) * 100
    return 100.0


def zeta_quantiles():
    pass


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data/corpus-build/text"))
    ap.add_argument("--dirs", nargs="+", default=None)
    ap.add_argument("--tok-cap", type=int, default=60_000_000)
    ap.add_argument("--tokenizer", type=Path, default=Path("data/qwen38-ple-fp8/tokenizer.json"))
    ap.add_argument("--out", type=Path, default=Path("probes/p2_report_v2.json"))
    args = ap.parse_args()

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(args.tokenizer))

    dirs = args.dirs or [d.name for d in args.root.iterdir() if d.is_dir()]
    results = {}
    mult = spec_from_weights("data/qwen38-ple-fp8")
    sizes = head_vocab_sizes()

    for dname in dirs:
        d = args.root / dname
        if not d.exists() or not list(d.glob("*.txt")):
            print(f"[skip] {dname}: no text", flush=True)
            continue
        tokens: list[int] = []
        for f in sorted(d.glob("*.txt")):
            txt = f.read_text(errors="ignore")
            tks = tok.encode(txt).ids
            tokens.extend(tks)
            if len(tokens) >= args.tok_cap:
                break
        tokens = tokens[: args.tok_cap]
        n = len(tokens)
        print(f"[{dname}] tokens={n}", flush=True)

        bigrams = Counter()
        trigrams = Counter()
        for i in range(2, n):
            bigrams[(tokens[i - 1], tokens[i])] += 1
            trigrams[(tokens[i - 2], tokens[i - 1], tokens[i])] += 1
        bg_sorted = sorted(bigrams.values(), reverse=True)
        tg_sorted = sorted(trigrams.values(), reverse=True)

        arr = np.array(tokens, dtype=np.int64)
        ids = ple_rowids(arr, mult, sizes)
        flat = ids.reshape(-1)
        rids = Counter(int(x) for x in flat)
        tier = {}
        counts_sorted = sorted(rids.values(), reverse=True)
        for top in (100, 1_000, 10_000, 100_000, 1_000_000):
            tier[str(top)] = round(sum(counts_sorted[:top]) / max(1, len(flat)) * 100, 3)
        head_uniq = [round(len(set(ids[:, h].tolist())) / max(1, ids.shape[0]) * 100, 2) for h in range(16)]

        results[dname] = {
            "tokens": n,
            "n_bigrams": len(bigrams),
            "n_trigrams": len(trigrams),
            "zipf_bigram": zeta_fit(bg_sorted),
            "zipf_trigram": zeta_fit(tg_sorted),
            "cov50_pct_types_bigram": cov_percent(bg_sorted, 0.5),
            "cov80_pct_types_bigram": cov_percent(bg_sorted, 0.8),
            "ple_unique_rows": len(rids),
            "ple_flat_unique_pct": round(len(rids) / len(flat) * 100, 2),
            "head_unique_pct": head_uniq,
            "tier_curve_top_rows": tier,
        }
        print(f"[{dname}] zipf(bg)={results[dname]['zipf_bigram']:.3f} flat-uniq={results[dname]['ple_flat_unique_pct']}% "
              f"top100={tier['100']}% top10k={tier['10000']}%", flush=True)

    args.out.parent.mkdir(exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1, ensure_ascii=False))
    print("report:", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def unused():
    zeta_quantiles()
