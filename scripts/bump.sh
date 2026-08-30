#!/usr/bin/env bash
# 版本 bump：统一更新 workspace 内所有 crate 与 Python 包版本，
# commit + 打 tag。用法: scripts/bump.sh <新版本如 0.2.0>
set -euo pipefail
cd "$(dirname "$0")/.."

V=$1
[[ "$V" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo "usage: bump.sh <MAJOR.MINOR.PATCH>"; exit 1; }

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
echo "bumped & tagged v${V} -> run 'cargo test' then 'git push && git push --tags' (release from GitHub UI)"
