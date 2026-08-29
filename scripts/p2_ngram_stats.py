#!/usr/bin/env python3
"""P2 v3：多域语料统计（阶段化 / 并行 / 可续跑 / uv 管理依赖）。

阶段（每个产物存在即跳过，天然断点续跑）：
  T1 tokenize : data/corpus-build/text/<domain>/*.txt → data/p2-work/tokens/<domain>/<file>.u32.npy
  T2 counters : 每域 bigram/trigram 唯一的 rank-coun (numpy 向量化)
  T3 rowids   : PLE rowid 分布 / 热集曲线（逐段 bincount 累加）
  T4 merge    : probes/p2_report_v2.json（含 agent_workload 摘要）

用法:
  uv run --project . python3 scripts/p2_ngram_stats.py --dirs fineweb zh agent \
        --workers 8 --tok-cap 30_000_000
"""
from __future__ import annotations
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from ref_ple_hash import ple_rowids, spec_from_weights, head_vocab_sizes

CHUNK_CHARS = 1_000_000
ROW_SPACE = 320_001_536


# ---------- T1：tokenize（chunk 级并行） ----------
def _encode_one(args):
    tok_path, text = args
    import tokenizers
    t = tokenizers.Tokenizer.from_file(tok_path)
    ids = t.encode(text).ids
    return np.array(ids, dtype=np.uint32)


def tokenize_domain(tok_path: str, domain: str, files, work: Path, workers: int, cap: int) -> Path:
    d = work / "tokens" / domain
    d.mkdir(parents=True, exist_ok=True)
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for fi, f in enumerate(files):
            out = d / f"{f.name}.u32.npy"
            if out.exists():
                done += 1
                continue
            with open(f, encoding="utf-8", errors="ignore") as fh:
                text = fh.read()
            chunks = [text[i:i + CHUNK_CHARS] for i in range(0, len(text), CHUNK_CHARS)]
            futs = [pool.submit(_encode_one, (str(tok_path), c)) for c in chunks]
            parts = [fut.result() for fut in futs]
            tokens = np.concatenate(parts) if parts else np.zeros(0, dtype=np.uint32)
            if len(tokens) > cap:
                tokens = tokens[:cap]
            np.save(out, tokens)
            done += 1
            print(f"  T1 [{domain}] {fi+1}/{len(files)} done={done} tok={len(tokens)} file={f.name}", flush=True)
    return d


# ---------- T2：bigram/trigram ----------
def _count_ngrams(tokens: np.ndarray):
    t = tokens.astype(np.uint64)
    # bigram: u64 = a<<32 | b
    k2 = (t[:-1] << 32) | t[1:]
    u2, c2 = np.unique(k2, return_counts=True)
    # trigram: u64 = a<<44 | b<<22 | c   (token < 2^22)
    k3 = (t[:-2] << 44) | (t[1:-1] << 22) | t[2:]
    u3, c3 = np.unique(k3, return_counts=True)
    return u2, c2, u3, c3


def _zipf(counts):
    if len(counts) < 2:
        return float("nan")
    xs = np.log(np.arange(1, len(counts) + 1))
    ys = np.log(np.maximum(counts, 1))
    try:
        return float(-np.polyfit(xs, ys, 1)[0])
    except Exception:
        return float("nan")


def stats_domain(tokens_dir: Path, work: Path, domain: str):
    out = work / "stats" / f"{domain}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        return json.loads(out.read_text())
    files = sorted(tokens_dir.glob("*.u32.npy"))
    total = 0
    counters = {}
    for f in files:
        t = np.load(f, mmap_mode="r")
        total += len(t)
    # 合并统计（fit 内存: 逐文件 vectorized + 分桶 merging 不可行 → 一次性拼接? cap 后 ~30M×4B=120MB 可拼)
    all_tokens = np.concatenate([np.load(f, mmap_mode="r").astype(np.uint64) for f in files])[:total]
    u2, c2, u3, c3 = _count_ngrams(all_tokens.astype(np.uint32))
    bg = c2.astype(np.float64)
    tg = c3.astype(np.float64)
    bg_sorted = np.sort(bg)[::-1]
    tg_sorted = np.sort(tg)[::-1]
    cov50 = np.searchsorted(np.cumsum(bg_sorted), bg_sorted.sum() * 0.5) / len(bg_sorted) * 100
    cov80 = np.searchsorted(np.cumsum(bg_sorted), bg_sorted.sum() * 0.8) / len(bg_sorted) * 100
    res = {
        "tokens": int(total),
        "n_bigrams": int(len(bg)),
        "n_trigrams": int(len(tg)),
        "zipf_bigram": _zipf(bg_sorted),
        "zipf_trigram": _zipf(tg_sorted),
        "cov50_pct_types_bigram": float(cov50),
        "cov80_pct_types_bigram": float(cov80),
    }
    out.write_text(json.dumps(res))
    print(f"  T2 [{domain}] tokens={total} bigrams={len(bg)} zipf={res['zipf_bigram']:.3f}", flush=True)
    return res


# ---------- T3：PLE rowid ----------
def rowid_domain(tokens_dir: Path, work: Path, domain: str, mult, sizes):
    out = work / "stats" / f"{domain}_rowid.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        return json.loads(out.read_text())
    counts = np.zeros(ROW_SPACE, dtype=np.uint32)
    files = sorted(tokens_dir.glob("*.u32.npy"))
    for f in files:
        t = np.load(f, mmap_mode="r")
        for i in range(0, len(t), 3_000_000):
            part = t[i:i + 3_000_000].astype(np.int64)
            ids = ple_rowids(part, mult, sizes).reshape(-1)
            counts += np.bincount(ids.astype(np.int64), minlength=ROW_SPACE).astype(np.uint32)
    nz = counts[counts > 0]
    total_q = int(nz.sum())
    top_idx = np.argpartition(counts, -1_000_000)[-1_000_000:]
    top = np.sort(counts[top_idx])[::-1]
    cum = np.cumsum(top)
    tier = {}
    for k in (100, 1_000, 10_000, 100_000, 1_000_000):
        kk = min(k, len(top))
        tier[str(k)] = round(float(cum[kk - 1]) / max(1, total_q) * 100, 3)
    res = {
        "unique_rows": int((counts > 0).sum()),
        "total_gets": total_q,
        "flat_unique_pct": round(int((counts > 0).sum()) / max(1, total_q) * 100, 2),
        "tier_curve_top_rows": tier,
    }
    out.write_text(json.dumps(res))
    print(f"  T3 [{domain}] unique={res['unique_rows']} flat-uniq={res['flat_unique_pct']}% tc={tier}", flush=True)
    return res


def main() -> int:
    from tokenizers import Tokenizer
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data/corpus-build/text"))
    ap.add_argument("--work", type=Path, default=Path("data/p2-work"))
    ap.add_argument("--dirs", nargs="+", default=None)
    ap.add_argument("--tok-cap", type=int, default=30_000_000)
    ap.add_argument("--tokenizer", type=Path, default=Path("data/qwen38-ple-fp8/tokenizer.json"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", type=Path, default=Path("probes/p2_report_v2.json"))
    args = ap.parse_args()

    tok_path = args.tokenizer
    Tokenizer.from_file(str(tok_path))  # 校验
    dirs = args.dirs or [d.name for d in args.root.iterdir() if d.is_dir()]
    mult = spec_from_weights("data/qwen38-ple-fp8")
    sizes = head_vocab_sizes()

    results = {}
    for dname in dirs:
        d = args.root / dname
        files = sorted(d.glob("*.txt"))
        if not files:
            print(f"[skip] {dname}", flush=True)
            continue
        print(f"[{dname}] T1 tokenize ({len(files)} files, workers={args.workers})", flush=True)
        td = tokenize_domain(str(tok_path), dname, files, args.work, args.workers, args.tok_cap)
        print(f"[{dname}] T2 counters", flush=True)
        results[dname] = stats_domain(td, args.work, dname)
        print(f"[{dname}] T3 PLE rowids", flush=True)
        results[dname].update(rowid_domain(td, args.work, dname, mult, sizes))

    args.out.parent.mkdir(exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1, ensure_ascii=False))
    print("report:", args.out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
