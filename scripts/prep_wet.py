#!/usr/bin/env python3
"""从 Common Crawl WARC.WET（gzip）流式抽取正文文本 → 语料目录 wet/*.txt。

WET 记录结构：头部 + 空行 + Content-Length 定长正文。
按规范：读头部到空行 → 取 content-length → 读定长字节。

用法: python3 scripts/prep_wet.py src --out <corpus/wet> --cap-mb 24
"""
import argparse
import gzip
import json
import time
from pathlib import Path


def wet_text_pages(fh, cap: int) -> bytes:
    out = bytearray()
    while len(out) < cap:
        headers = bytearray()
        while True:
            line = fh.readline()
            if not line:
                return bytes(out)
            if line in (b"\r\n", b"\n", b"\r", b""):
                break
            headers += line
        clen = 0
        is_warcinfo = False
        for h in headers.split(b"\n"):
            if h.lower().startswith(b"content-length:"):
                try:
                    clen = int(h.split(b":")[1].strip())
                except ValueError:
                    clen = 0
            if h.lower().startswith(b"warc-type: warcinfo"):
                is_warcinfo = True
        if is_warcinfo:
            fh.read(clen if clen else 1 << 16)
            continue
        if clen <= 0 or clen > (1 << 30):
            continue
        data = fh.read(clen)
        if not data:
            return bytes(out)
        out += data
    return bytes(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path, default=Path("/Volumes/My Passport/datasets/commoncrawl"))
    ap.add_argument("--out", type=Path, default=Path("data/qwen38-ple-fp8/corpus/wet"))
    ap.add_argument("--cap-mb", type=float, default=24.0)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    files = sorted(f for f in args.src.rglob("*.warc.wet.gz") if not f.name.startswith("._"))
    total_mb = sum(f.stat().st_size for f in files) / 1e6
    print(f"found {len(files)} WET files, {total_mb:.0f} MB compressed")
    if args.dry:
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    cap = int(args.cap_mb * 1_000_000)
    manifest = []
    t0 = time.time()
    for fi, f in enumerate(files):
        with gzip.open(f, "rb") as gz:
            text = wet_text_pages(gz, cap)
        if not text:
            continue
        chunk = 16 << 20
        n_chunks = (len(text) + chunk - 1) // chunk
        for ci in range(n_chunks):
            part = text[ci * chunk:(ci + 1) * chunk]
            (args.out / f"cc{fi:02d}_{ci:02d}.txt").write_bytes(part)
        mb = len(text) / 1e6
        manifest.append({"file": f.name, "cap_mb": round(mb, 1), "n_chunks": n_chunks})
        print(f"{fi:2d}/{len(files)} -> {mb:.1f} MB text ({n_chunks} chunks)", flush=True)
    (args.out.parent / "wet-manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"done in {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
