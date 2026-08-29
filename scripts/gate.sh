#!/usr/bin/env bash
# 本地门禁 gate：code gate（必过）+ bench gate（数据存在时）
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== [gate] fmt =="
cargo fmt --all --check

echo "== [gate] clippy (-D warnings) =="
cargo clippy --all-targets --all-features -- -D warnings

echo "== [gate] cargo test --workspace =="
cargo test --workspace

echo "== [gate] bench gate (数据存在才跑) =="
if [[ -d data/real-rows ]]; then
  echo "  [gate] 真表存在 -> P1 gather / P4 view 基准（人工核验 ≥1M 行/s 与 1.6x 放大）"
  echo "  [gate] TODO(phase3): 固定参数自动化 gate 接入 p4view"
else
  echo "  [gate] 无真表数据：跳过基准（CI 同理）"
fi

echo "== [gate] PASS =="
