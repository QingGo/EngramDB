#!/usr/bin/env python3
"""高质语料库构建（download + parse 一体，可断点续传）。

域：
- fineweb : HuggingFaceFW/fineweb-edu  1 片 (CC-MAIN-2013-20/train-00000-of-00014.parquet, 2.37GB)
            → 清洗后英文 edu 语料；文本采样 cap 150MB
- zh      : opencsg/chinese-fineweb-edu  Skypile/00000.parquet (779MB) → cap 100MB
- agent   : trace-commons/agent-traces   data/train-00000-of-00001.parquet (66MB) → 全量文本
- cctraces: semianalysisai/cc-traces-weka-no-subagents-051226 traces.jsonl (2.64GB)
            → 不存文本，提炼 requests 统计 → probes/agent_workload_stats.json

下载路由（auto）：modelscope API 直连 / hf-mirror 直连 / huggingface.co+代理7897；逐候选探测，谁速率高且通就用谁；
用 `curl -L -C -` 支持断点续传。所有产物在 data/corpus-build/ 下（非 git）。

用法: python3 scripts/corpus_build.py            # 全流程（下载+解析+统计）
      python3 scripts/corpus_build.py --only-dl  # 只下载
"""
import argparse
import json
import logging
import os
import shutil
import time
import urllib.request
from pathlib import Path

log = logging.getLogger("corpus")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

ROOT = Path("./data/corpus-build")
RAW = ROOT / "raw"
TXT = ROOT / "text"
PROBES = Path("probes")

PROXY = "http://127.0.0.1:7897"
HF_MIRROR = "https://hf-mirror.com"
PROXY_HF = PROXY  # 经 huggingface.co 走代理

def curl_cmd(url: str, out: Path, resume: bool = True, proxy: str | None = None, via_proxy_host: str | None = None):
    u = url
    if via_proxy_host is not None and via_proxy_host != HF_MIRROR:
        u = url  # url 已是 huggingface.co 形态
    cmd = ["curl", "-L"]
    if resume and out.exists():
        cmd += ["-C", "-"]
    if proxy and via_proxy_host is None:
        cmd += ["-x", proxy]
    elif via_proxy_host == HF_MIRROR and proxy:
        cmd += ["-x", proxy]
    cmd += ["-o", str(out), "--retry", "3", u]
    return cmd

def probe(url: str, proxy: str | None = None, seconds: int = 8):
    """限时探速（约 2MB），返回 bytes/s，失败 0。"""
    try:
        req = urllib.request.Request(
            url, headers={"Range": "bytes=0-2097151",
                          "User-Agent": "engramdb-corpus/0.1"})
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({"http": proxy, "https": proxy}) if proxy else urllib.request.ProxyHandler({}))
        t0 = time.time()
        with opener.open(req, timeout=seconds) as r:
            chunks = 0
            buf = r.read(1 << 16)
            while buf and time.time() - t0 < seconds and chunks < (4 << 20):
                chunks += len(buf)
                buf = r.read(1 << 16)
            # 需要读到 Range 字节直到服务器关闭
        dt = max(time.time() - t0, 0.05)
        return chunks / dt
    except Exception:
        return 0.0

def pick_route(candidates):
    best, best_speed = None, 0.0
    for name, url, proxy, via in candidates:
        s = probe(url, proxy=proxy)
        log.info("route %-14s %6.2f MB/s", name, s / 1e6)
        if s > best_speed:
            best_speed, best = s, (name, url, proxy, via)
    if best is None:
        raise RuntimeError("no route available")
    return best

def download_auto(file_key, routes, out: Path, expected_seconds: float = 1.0):
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size > 1000:
        log.info("%s already present (%d MB), skip dl", file_key, out.stat().st_size // 1048576)
        return
    name, url, proxy, via = pick_route(routes)
    log.info("downloading %s via %s -> %s", file_key, name, out.name)
    subprocess_dl(url, out, proxy, via)
    sz = out.stat().st_size
    log.info("%s done: %.2f MB", file_key, sz / 1e6)

def subprocess_dl(url, out, proxy, _via):
    import subprocess
    cmd = ["curl", "-L", "-C", "-", "--retry", "5", "--retry-all-errors", "-o", str(out)]
    if proxy:
        cmd += ["-x", proxy]  # 仅 proxy-hf 路由携带
    cmd.append(url)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 and not (out.exists() and out.stat().st_size > 0):
        raise RuntimeError(f"curl failed ({r.returncode}): {r.stderr[-400:]}")

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only-dl", action="store_true")
    args = ap.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    TXT.mkdir(parents=True, exist_ok=True)

    # ---- 下载清单 ----
    dl_list = [
        # fineweb（modelscope 镜像 = HF repo id）
        {
            "key": "fineweb", "out": RAW / "fineweb.parquet", "route_key": "fineweb",
            "routes": [
                ("modelscope", f"https://modelscope.cn/api/v1/datasets/HuggingFaceFW/fineweb-edu/repo?Revision=master&FilePath=data/CC-MAIN-2013-20/train-00000-of-00014.parquet", None, False),
                ("hf-mirror", f"{HF_MIRROR}/datasets/HuggingFaceFW/fineweb-edu/resolve/main/data/CC-MAIN-2013-20/train-00000-of-00014.parquet", None, True),
                ("proxy-hf", f"https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/data/CC-MAIN-2013-20/train-00000-of-00014.parquet", PROXY, False),
            ],
        },
        # zh fineweb-edu（modelscope 原生）
        {
            "key": "zh", "out": RAW / "zh_skypile.parquet", "route_key": "zh",
            "routes": [
                ("modelscope", f"https://modelscope.cn/api/v1/datasets/opencsg/chinese-fineweb-edu/repo?Revision=master&FilePath=Skypile/00000.parquet", None, False),
                ("hf-mirror-zh", f"{HF_MIRROR}/datasets/opencsg/chinese-fineweb-edu/resolve/main/Skypile/00000.parquet", None, True),
            ],
        },
        # agent traces（文本）
        {
            "key": "agent", "out": RAW / "agent_traces.parquet", "route_key": "agent",
            "routes": [
                ("proxy-hf", f"https://huggingface.co/datasets/trace-commons/agent-traces/resolve/main/data/train-00000-of-00001.parquet", PROXY, False),
                ("hf-mirror", f"{HF_MIRROR}/datasets/trace-commons/agent-traces/resolve/main/data/train-00000-of-00001.parquet", None, True),
            ],
        },
        # cc-traces 统计（jsonl 2.6GB）
        {
            "key": "cctraces", "out": RAW / "cc_traces.jsonl", "route_key": "cctraces",
            "routes": [
                ("proxy-hf", f"https://huggingface.co/datasets/semianalysisai/cc-traces-weka-no-subagents-051226/resolve/main/traces.jsonl", PROXY, False),
                ("hf-mirror", f"{HF_MIRROR}/datasets/semianalysisai/cc-traces-weka-no-subagents-051226/resolve/main/traces.jsonl", None, True),
            ],
        },
    ]
    for item in dl_list:
        download_auto(item["key"], item["routes"], item["out"])

    if args.only_dl:
        log.info("downloads only done")
        return 0

    # ---- 解析 ----
    parse_fineweb(RAW / "fineweb.parquet")
    parse_zh(RAW / "zh_skypile.parquet")
    parse_agent(RAW / "agent_traces.parquet")
    parse_cctraces(RAW / "cc_traces.jsonl")

    manifest = {"built_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    for d in sorted(TXT.iterdir()):
        if d.is_dir():
            mb = sum(f.stat().st_size for f in d.glob("*.txt")) / 1e6
            manifest[f"text/{d.name}"] = round(mb, 2)
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    log.info("manifest: %s", manifest)
    return 0

def parse_fineweb(p: Path):
    if (TXT / "fineweb" / "done").exists():
        return
    import pyarrow.parquet as pq
    out = TXT / "fineweb"; out.mkdir(parents=True, exist_ok=True)
    cap = 150_000_000
    written, w = 0, open(out / "fineweb.txt", "w", encoding="utf-8", errors="ignore")
    for batch in pq.ParquetFile(p).iter_batches(batch_size=20000, columns=["text"]):
        if written >= cap:
            break
        for s in batch.column("text").to_pylist():
            t = s or ""
            if len(t) < 200:
                continue
            w.write(t.strip().replace("\x00", "")); w.write("\n")
            written += len(t)
            if written >= cap:
                break
    w.close()
    (out / "done").write_text(f"{written}")
    log.info("fineweb text: %d MB", written / 1e6)

def parse_zh(p: Path):
    if (TXT / "zh" / "done").exists():
        return
    import pyarrow.parquet as pq
    out = TXT / "zh"; out.mkdir(parents=True, exist_ok=True)
    f = pq.ParquetFile(p)
    col = "text" if "text" in f.schema.names else "content"
    cap = 100_000_000
    written, w = 0, open(out / "zh.txt", "w", encoding="utf-8", errors="ignore")
    for batch in f.iter_batches(batch_size=20000, columns=[col]):
        if written >= cap:
            break
        for s in batch.column(col).to_pylist():
            t = s or ""
            if len(t) < 100:
                continue
            w.write(t.strip().replace("\x00", "")); w.write("\n")
            written += len(t)
            if written >= cap:
                break
    w.close()
    (out / "done").write_text(f"{written}")
    log.info("zh text: %d MB", written / 1e6)

def parse_agent(p: Path):
    if (TXT / "agent" / "done").exists():
        return
    import pyarrow.parquet as pq
    out = TXT / "agent"; out.mkdir(parents=True, exist_ok=True)
    # 分 harness 组织
    import hashlib
    cap = 66_000_000
    written, per = 0, {}
    t = pq.ParquetFile(p).read()
    rows = t.num_rows
    for i in range(rows):
        row = t.slice(i, 1)
        harness = row.column("harness")[0].as_py() or "other"
        messages = row.column("messages")[0].as_py()
        if messages is None:
            messages = row.column("trace")[0].as_py() if "trace" in t.column_names else None
        text = flatten_messages(messages)
        if not text or len(text) < 300:
            continue
        f = per.setdefault(harness, open(out / f"{harness}.txt", "w", encoding="utf-8", errors="ignore"))
        f.write(text); f.write("\n\n--- session ---\n\n")
        written += len(text)
        if written >= cap:
            break
    for f in per.values():
        f.close()
    (out / "done").write_text(f"{written}")
    log.info("agent text: %d MB", written / 1e6)

def flatten_messages(msgs) -> str:
    def fmt(msg):
        role = msg.get("role", "?")
        content = msg.get("content")
        if isinstance(content, list):
            parts = []
            for c in content:
                if isinstance(c, dict):
                    if c.get("type") == "text":
                        parts.append(c.get("text") or "")
                    elif c.get("type") == "tool_use":
                        parts.append(f"[tool_use {c.get('name')}]\n{json.dumps(c.get('input', {}), ensure_ascii=False)[:2000]}")
                    elif c.get("type") == "tool_result":
                        parts.append(str(c.get("content") or "")[:2000])
                    else:
                        parts.append(str(c)[:1000])
                else:
                    parts.append(str(c)[:1000])
            text = "\n".join(parts)
        else:
            text = content if isinstance(content, str) else str(content or "")
        return f"<{role}>\n{text}"
    out = []
    for m in msgs or []:
        if isinstance(m, dict):
            out.append(fmt(m))
    return "\n".join(out)

def parse_cctraces(p: Path):
    if PROBES.joinpath("agent_workload_stats.json").exists():
        return
    import re
    stats = {"sessions": 0, "requests": 0, "in_tokens": [], "out_tokens": [],
             "ttft": [], "api_time": [], "think_time": [], "models": {},
             "blocks_per_request": []}
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            stats["sessions"] += 1
            for r in obj.get("requests", []):
                stats["requests"] += 1
                stats["in_tokens"].append(r.get("in", 0))
                stats["out_tokens"].append(r.get("out", 0))
                if r.get("ttft") is not None:
                    stats["ttft"].append(r["ttft"])
                if r.get("api_time") is not None:
                    stats["api_time"].append(r["api_time"])
                if r.get("think_time") is not None:
                    stats["think_time"].append(r["think_time"])
                stats["blocks_per_request"].append(len(r.get("hash_ids", [])))
                stats["models"][r.get("model", "?")] = stats["models"].get(r.get("model", "?"), 0) + 1
    agg = {k: (v if isinstance(v, int) else {"n": len(v),
            "mean": round(sum(x for x in v if x is not None) / max(1, len([x for x in v if x is not None])), 2),
            "p50": rounded_quantile(v, 0.5), "p95": rounded_quantile(v, 0.95),
            "max": max(v) if v else None}) for k, v in stats.items() if k not in ("models",) or True}
    out = {"sessions": agg["sessions"], "requests": agg["requests"],
           "in_tokens": agg["in_tokens"], "out_tokens": agg["out_tokens"],
           "ttft": agg["ttft"], "api_time": agg["api_time"], "think_time": agg["think_time"],
           "blocks_per_request": agg["blocks_per_request"], "models": stats["models"]}
    PROBES.mkdir(exist_ok=True)
    (PROBES / "agent_workload_stats.json").write_text(json.dumps(out, indent=1))
    log.info("agent workload stats: %d sessions / %d requests", out["sessions"], out["requests"])

def rounded_quantile(xs, q):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    i = min(int(q * (len(xs) - 1)), len(xs) - 1)
    return round(xs[i], 3)

def _rm(p: Path):
    if p.is_dir():
        shutil.rmtree(p, ignore_errors=True)

if __name__ == "__main__":
    raise SystemExit(main())
