#!/usr/bin/env bash
# 版本 bump：把 workspace 全部公开 crate 的 version 统一更新（当前各 crate 独立 version 字段），
# commit + 打 tag。用法: scripts/bump.sh <新版本如 0.2.0>
set -euo pipefail
cd "$(dirname "$0")/.."

V=$1
[[ "$V" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo "usage: bump.sh <MAJOR.MINOR.PATCH>"; exit 1; }

for c in engramdb-keygen engramdb-core engramdb-io engramdb; do
  python3 - "$c" "$V" <<'EOF'
import re, sys
c, v = sys.argv[1], sys.argv[2]
p = f"crates/{c}/Cargo.toml"
s = open(p).read()
s = re.sub(r'^(version = )"[^"]+"', rf'\g<1>"{v}"', s, count=1, flags=re.M)
open(p, "w").write(s)
print(f"  {c}: v{v}")
EOF
done

git add -A
git commit -m "release: bump v${V}"
git tag -a "v${V}" -m "EngramDB v${V}"
echo "bumped & tagged v${V} -> run 'cargo test' then 'git push && git push --tags' (release from GitHub UI)"
