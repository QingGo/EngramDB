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
old, new = sys.argv[1], sys.argv[2]
crates = [
    "engramdb-keygen", "engramdb-core", "engramdb-io", "engramdb",
    "engramdb-bench", "engramdb-python", "engramdb-pyo3",
]
paths = ["Cargo.toml"] + [f"crates/{c}/Cargo.toml" for c in crates] + [
    "python/pyproject.toml",
    "python/engramdb/__init__.py",
]
for p in paths:
    s = open(p).read()
    s2 = s.replace(f'version = "{old}"', f'version = "{new}"')
    s2 = s2.replace(f'__version__ = "{old}"', f'__version__ = "{new}"')
    if s2 != s:
        open(p, "w").write(s2)
        print(f"  {p}: {old} -> {new}")
PY

git add -A
git commit -m "release: bump v${V}"
git tag -a "v${V}" -m "EngramDB v${V}"
if [[ "$SKIP_GATE" == "1" ]]; then
  echo "bumped & tagged v${V} -> run 'bash scripts/release_gate.sh' before pushing"
else
  echo "bumped & tagged v${V} -> release gate already passed; push with 'git push && git push --tags' (release from GitHub UI)"
fi
