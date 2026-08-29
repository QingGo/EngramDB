#!/usr/bin/env bash
# 发布脚本：按依赖顺序 publish workspace 的公开 crates（供 CI 与手动执行）。
# 前提：CARGO_REGISTRY_TOKEN 已注入（CI secrets 或 ~/.cargo/credentials.toml）。
# 注意：发布期间若本机有 registry 替换（如 mirrors），会因 index 未同步而失败——
# CI（干净环境）无此问题；手动执行请先临时移除替换源（见 docs/session-log.md）。
set -euo pipefail
cd "$(dirname "$0")/.."

CRATES=(engramdb-keygen engramdb-core engramdb-io engramdb)

# tag 与 manifest 版本一致性校验（本地/CI 均校验）
if [[ -n "${RELEASE_TAG:-}" ]]; then
  want="${RELEASE_TAG#v}"
  for c in "${CRATES[@]}"; do
    have=$(grep -m1 '^version = ' "crates/$c/Cargo.toml" | cut -d'"' -f2)
    [[ "$have" == "$want" ]] || { echo "  mismatch: $c manifest v$have != tag $want"; exit 1; }
  done
  echo ">> tag $RELEASE_TAG matches all manifests"
fi

for c in "${CRATES[@]}"; do
  echo ">> cargo publish $c"
  cargo publish -p "$c" --registry crates-io
done
echo ">> all crates published"
