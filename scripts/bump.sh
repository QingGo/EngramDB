#!/usr/bin/env bash
# 版本 bump：统一更新 workspace 内所有 crate 与 Python 包版本，
# commit + 打 tag。用法: scripts/bump.sh [--skip-gate] <新版本如 0.2.0>
#
# 默认先跑 scripts/release_gate.sh；若只需要快速打 tag，可显式加 --skip-gate
# 或设置 ENGRAMDB_SKIP_GATE=1。
set -euo pipefail
cd "$(dirname "$0")/.."

SKIP_GATE="${ENGRAMDB_SKIP_GATE:-0}"
if [[ "${1:-}" == "--skip-gate" ]]; then
  SKIP_GATE=1
  shift
fi

V=${1:-}
if [[ -z "$V" ]]; then
  echo "usage: bump.sh [--skip-gate] <MAJOR.MINOR.PATCH>"
  exit 1
fi
[[ "$V" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo "usage: bump.sh [--skip-gate] <MAJOR.MINOR.PATCH>"; exit 1; }

if [[ "$SKIP_GATE" != "1" ]]; then
  echo ">> running release gate before bump (set ENGRAMDB_SKIP_GATE=1 to skip) ..."
  bash scripts/release_gate.sh
fi

OLD=$(grep -m1 '^version = ' Cargo.toml | cut -d'"' -f2)
if [[ -z "$OLD" || "$OLD" == "$V" ]]; then
  echo "cannot determine previous version (or already at $V)"; exit 1
fi

python3 - "$OLD" "$V" <<'PY'
import sys
from pathlib import Path
old, new = sys.argv[1], sys.argv[2]
crates = [
    "engramdb-keygen", "engramdb-core", "engramdb-io", "engramdb",
    "engramdb-bench", "engramdb-cabi", "engramdb-pyo3",
]
paths = ["Cargo.toml"] + [f"crates/{c}/Cargo.toml" for c in crates] + [
    "python/pyproject.toml",
    "python/engramdb/__init__.py",
]
# 先整体校验路径存在：crate 改名时这里漏改过一次，脚本以一条
# FileNotFoundError traceback 崩掉，很难看出是"名单过期"而不是真的文件丢失。
missing = [p for p in paths if not Path(p).is_file()]
if missing:
    sys.exit(
        "bump.sh: 版本文件清单里有不存在的路径（多半是 crate 改名后忘了同步本脚本）：\n  "
        + "\n  ".join(missing)
    )
touched = 0
for p in paths:
    s = Path(p).read_text()
    s2 = s.replace(f'version = "{old}"', f'version = "{new}"')
    s2 = s2.replace(f'__version__ = "{old}"', f'__version__ = "{new}"')
    if s2 != s:
        Path(p).write_text(s2)
        touched += 1
        print(f"  {p}: {old} -> {new}")

# README 的版本标记是 `**vX.Y.Z**` 格式（不是 `version = "X.Y.Z"`），所以上面那圈
# 替换覆盖不到它。曾经本脚本完全不管 README，于是 v0.3.0 发布后 README 仍写着 v0.2.12。
# 这里**硬失败**而不是静默略过：README 停在旧版本号正是"用户看到的第一句话是错的"。
readme = Path("README.md")
rs = readme.read_text()
marker_old = f"**v{old}**"
if marker_old not in rs:
    sys.exit(
        f"bump.sh: README.md 里找不到版本标记 {marker_old} —— 拒绝打 tag。\n"
        "  README 顶部/状态表里的版本号必须随发布更新；找不到标记说明格式变了，请同步本脚本。"
    )
readme.write_text(rs.replace(marker_old, f"**v{new}**"))
print(f"  README.md: {marker_old} -> **v{new}**")
touched += 1
if touched == 0:
    sys.exit(f"bump.sh: 没有任何文件包含版本 {old} —— 拒绝打 tag")
PY

git add -A
git commit -m "release: bump v${V}"
git tag -a "v${V}" -m "EngramDB v${V}"
if [[ "$SKIP_GATE" == "1" ]]; then
  echo "bumped & tagged v${V} -> run 'bash scripts/release_gate.sh' before pushing"
else
  echo "bumped & tagged v${V} -> release gate already passed; push with 'git push && git push --tags' (release from GitHub UI)"
fi
