#!/usr/bin/env bash
# 发布前完整门禁（release gate）
#
# 用途：在执行 scripts/bump.sh / 推送 tag 之前，本地一次性跑完所有应过的门槛。
# 等价于 CI 中 release/publish/release-assets 的 preflight + Python 冒烟，
# 避免“推到 GitHub 才发现 CI 失败”。
#
# 用法:
#   bash scripts/release_gate.sh
#   PYTHON=/usr/local/bin/python3.14 bash scripts/release_gate.sh
#
# 可选环境变量:
#   PYTHON      用于运行 Python 冒烟脚本的解释器（默认 python3）
#   SKIP_BENCH  设为 1 时跳过真表 bench（仅在没有真表/需要快速门禁时使用）
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3}"
ROOT="$(pwd)"

# 与 build_pyo3.sh / build_wheel.sh 保持一致的沙箱/受限环境兼容：
# 如果用户的 ~/.cargo 不可写（例如 CI 只读缓存），自动切到 /tmp/cargo-home。
if [[ -z "${CARGO_HOME:-}" && ! -w "$HOME/.cargo" ]]; then
  export CARGO_HOME="/tmp/cargo-home"
  echo "note: ~/.cargo is not writable; using CARGO_HOME=$CARGO_HOME"
fi

echo "=============================================="
echo "EngramDB release gate"
echo "  python = $PYTHON"
echo "  root   = $ROOT"
echo "  cargo  = ${CARGO_HOME:-$HOME/.cargo}"
echo "=============================================="

echo
echo "== [release-gate] cargo fmt =="
cargo fmt --all --check

echo
echo "== [release-gate] cargo clippy (-D warnings) =="
cargo clippy --all-targets --all-features -- -D warnings

echo
echo "== [release-gate] cargo test --workspace =="
cargo test --workspace

if [[ "${SKIP_BENCH:-0}" != "1" && -d data/real-rows && -f probes/view-keys-20k.txt ]]; then
  echo
  echo "== [release-gate] bench gate (real rows available) =="
  bash scripts/gate.sh
else
  echo
  echo "== [release-gate] bench gate skipped =="
  if [[ -d data/real-rows ]]; then
    echo "  (data/real-rows exists, but SKIP_BENCH or probes/view-keys-20k.txt missing)"
  else
    echo "  (data/real-rows not present; same as CI)"
  fi
fi

if [[ -d data/real-rows ]]; then
  echo
  echo "== [release-gate] real Arrow IPC smoke =="
  PYTHONPATH=python "$PYTHON" scripts/real_arrow_smoke.py
  echo
  echo "== [release-gate] real serving perf thresholds =="
  PYTHONPATH=python "$PYTHON" scripts/real_perf_gate.py
fi

echo
echo "== [release-gate] build PyO3 native extension =="
bash scripts/build_pyo3.sh

echo
echo "== [release-gate] build C ABI cdylib =="
cargo build --release -p engramdb-python

echo
echo "== [release-gate] python wheel smoke =="
PYTHONPATH=python "$PYTHON" scripts/python_wheel_smoke.py

echo
echo "== [release-gate] service smoke =="
PYTHONPATH=python "$PYTHON" scripts/service_smoke.py

echo
echo "== [release-gate] C ABI smoke =="
PYTHONPATH=python "$PYTHON" scripts/c_abi_smoke.py

echo
echo "== [release-gate] decode baseline check =="
if [[ -f probes/cpu_tiny_baseline.csv || -f probes/qwen35_cpu_baseline.csv ]]; then
  PYTHONPATH=python "$PYTHON" scripts/decode_baseline_check.py
else
  echo "  baseline CSVs missing; skipping (same as CI)"
fi

echo
echo "=============================================="
echo "RELEASE_GATE_OK"
echo "=============================================="
