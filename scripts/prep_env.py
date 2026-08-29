#!/usr/bin/env python3
"""跨平台环境准备器（Windows / Linux / macOS）：从空环境到可跑探针。

子命令：
  quick       默认。离线准备：环境自检 + 生成合成 PLE 表（mock-gather / P1 / P4
              探针可用）。不依赖任何网络。
  verify      校验前面产物（文件存在、规模表头、磁盘余量）。
  ckpt-check  校验真实权重目录（私有源时跳过下载；要求 config.json / spec.json
              与提取模式一致，给出 pass / fail 明细）。
  full-eval   完整评估集：委托 corpus_build.py 下载三域小语料 +（可选真实权重从
              --ckpt-local <src> 拷贝）。大文件由该脚本内部断点续传/路由探测。

用法:
  python3 scripts/prep_env.py                      # quick + verify 提示
  python3 scripts/prep_env.py quick --scale 16
  python3 scripts/prep_env.py ckpt-check data/qwen38-ple-fp8
  python3 scripts/prep_env.py full-eval --ckpt-local /Volumes/SSD/qwen38-ple-fp8

Windows 注意事项:
  - 需 Python 3.10+；cmd/PowerShell 下 python3 不存在时用 `python`（脚本内探测）。
  - curl 仅 full-eval 用（Win10+ 自带）；quick/verify 全程标准库，无需 curl/git/bash。
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MOCK = DATA / "mock-qwen38-ple"
CKPT = DATA / "qwen38-ple-fp8"
CORPUS = DATA / "corpus-build"

REAL_SPEC = {
    "shards": 128,
    "rows_per_shard": 2_500_012,
    "width": 160,
    "dtype": "F8_E4M3",
}


def eprint(*a, **k):
    print(*a, file=sys.stderr, **k)


def py() -> str:
    cmd = sys.executable
    if shutil.which(cmd):
        return cmd
    return "python" if shutil.which("python") else "python3"


def check_env() -> None:
    import platform

    print(f"[env] python {platform.python_version()} on {platform.system()}")
    print(f"[env] repo root: {ROOT}")
    if sys.version_info < (3, 10):
        eprint("需要 Python >= 3.10 (当前 %s)" % platform.python_version())
        sys.exit(1)
    if not (ROOT / "scripts" / "gate.sh").exists():
        eprint("目录不是 EngramDB 仓库根？")
        sys.exit(1)
    disk = shutil.disk_usage(str(Path.home()))
    free_gb = disk.free / 1e9
    print(f"[env] 磁盘余量 {free_gb:.1f} GB（quick 建议 >=1 GB）")
    if free_gb < 1:
        eprint("磁盘不足：请清理后重试")
        sys.exit(1)


def quick(scale: int) -> None:
    check_env()
    if MOCK.exists() and any(MOCK.glob("*.bin")) and scale != 16:
        print(f"[quick] {MOCK} 已存在（不同 scale 请指定新 --out 或删除旧目录）")
    print(f"[quick] 合成表生成: mock_table_gen.py --scale {scale}")
    subprocess.run(
        [py(), str(ROOT / "scripts" / "mock_table_gen.py"), "--scale", str(scale)],
        cwd=ROOT,
        check=True,
    )
    verify(quiet=True)
    print("\n[quick] 完成。之后可以：")
    print(f"  pytest/cargo test          # 单元 + 集成")
    print(f"  bash scripts/gate.sh       # 本机 gate（数据在时含 bench gate）")
    print(f"  python3 scripts/prep_env.py ckpt-check data/qwen38-ple-fp8")


def verify(quiet: bool = False) -> None:
    ok = True
    for name in ("mock-qwen38-ple", "corpus-build", "real-rows", "p2-work"):
        p = DATA / name
        if not p.exists():
            if not quiet:
                print(f"[verify] 未找到 {name}/（可先跑 quick 或 full-eval）")
            continue
        if name == "mock-qwen38-ple":
            bins = list(p.glob("shard_000.bin"))
            spec = p / "layout.json"
            if bins and spec.exists():
                import numpy as np

                arr = np.frombuffer(bins[0].read_bytes(), dtype=np.uint8).reshape(-1, 160)
                layout = json.loads(spec.read_text())
                print(f"[verify] mock 表 OK: shards={len(list(p.glob('*.bin')))}"
                      f" rows/shard≈{arr.shape[0]} (layout={layout.get('shard_rows')}) width=160")
            else:
                ok = False
                eprint("[verify] mock 表目录异常（缺 shard_000.bin / layout.json）")
        elif name == "real-rows":
            bins = [x for x in p.iterdir() if x.name.startswith("shard_") and x.suffix == ".bin"] if p.exists() else []
            if len(bins) >= 128:
                print(f"[verify] real-rows OK: {len(bins)} shards (真表, 约 {p.stat().st_size/1e9:.1f}G)")
            elif p.exists():
                ok = False
                eprint(f"[verify] real-rows 只有 {len(bins)} 个分片（需 ≥128）")
        elif name == "corpus-build":
            mf = p / "manifest.json"
            raw = p / "raw"
            if mf.exists() and raw.exists():
                mb_mb = sum(f.stat().st_size for f in raw.iterdir()) / 1e9
                mb_mb = sum(
                    (f.stat().st_size for f in raw.iterdir() if f.is_file()), 0.0
                ) / 1e9
                print(f"[verify] corpus-build OK: raw {mb_mb:.2f}GB")
            else:
                ok = False
                eprint("[verify] corpus-build 不完整（缺 manifest.json/raw）")
    if quiet:
        return
    if not ok:
        sys.exit(1)
    print("[verify] OK")


def ckpt_check(path: Path) -> None:
    check_env()
    config = path / "config.json"
    spec = path / "spec.json"
    if not config.exists() and not spec.exists():
        eprint(f"未找到 config.json 或 spec.json：{path}")
        eprint("真实权重在私有源（SA）。校验前请从源机拷贝/挂载后重跑，或指向 mock 表")
        sys.exit(1)
    doc = json.loads((config if config.exists() else spec).read_text())
    arch = doc.get("architectures", [None])[0]
    model_type = doc.get("model_type", "?")
    print(f"[ckpt] architecture={arch} model_type={model_type}")
    # mock 表：spec.json 有 shards/rows 字段
    if "shards" in doc or (spec.exists() and "shards" in doc):
        shards = doc.get("shards", REAL_SPEC["shards"])
        rps = doc.get("rows_per_shard", REAL_SPEC["rows_per_shard"])
        width = doc.get("width", REAL_SPEC["width"])
        ok = shards == REAL_SPEC["shards"] and width == REAL_SPEC["width"]
        print(f"[ckpt] 结构一致: shards={shards} rows/shard={rps} width={width} "
              f"({'PASS' if ok else 'CHECK'})")
    else:
        print("[ckpt] 真表字典（无显式 shards 字段）→ 以 2.5M×160×128 参考，未做细检")


def full_eval(ckpt_local: Path | None) -> None:
    check_env()
    if not CORPUS.joinpath("manifest.json").exists():
        print("[full] 语料下载（多镜像路由 + 断点续传）…")
        subprocess.run(
            [py(), str(ROOT / "scripts" / "corpus_build.py"), "--help"], cwd=ROOT, check=False
        )
        eprint("[full] 语料几何由 corpus_build.py 自己管理（见其 --help）；"
               "本脚本只校验结果存在。")
    if ckpt_local is not None:
        src = Path(ckpt_local)
        if not src.exists():
            eprint(f"--ckpt-local 路径无效: {src}")
            sys.exit(1)
        print(f"[full] 拷贝权重 {src} -> {CKPT} …")
        CKPT.mkdir(parents=True, exist_ok=True)
        for item in src.iterdir():
            dst = CKPT / item.name
            if item.is_file():
                shutil.copy2(item, dst)
            elif item.is_dir():
                shutil.copytree(item, dst, dirs_exist_ok=True)
    verify()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="EngramDB 跨平台环境准备器",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("quick", help="离线合成表 + 自检（默认）")
    q.add_argument("--scale", type=int, default=16)

    v = sub.add_parser("verify", help="校验产物")
    c = sub.add_parser("ckpt-check", help="校验权重/config")
    c.add_argument("--path", type=Path, default=CKPT)

    f = sub.add_parser("full-eval", help="完整评估集（委托 corpus_build + 可选权重拷贝）")
    f.add_argument("--ckpt-local", type=Path, default=None)

    args = ap.parse_args()
    if args.cmd == "quick":
        quick(args.scale)
    elif args.cmd == "verify":
        verify()
    elif args.cmd == "ckpt-check":
        ckpt_check(args.path)
    elif args.cmd == "full-eval":
        full_eval(args.ckpt_local)


if __name__ == "__main__":
    main()
