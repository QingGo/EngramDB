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

echo "== [gate] bench gate (真表存在才跑) =="
if [[ -d data/real-rows ]]; then
  if [[ -f probes/view-keys-20k.txt ]]; then
    V=/tmp/mtrl-gate-view.bin
    rm -f "$V" probes/view-manifest.json
    echo "  [gate] build 20k view (2560B 紧凑槽) ..."
    if ! target/release/p4view build data/real-rows 20000 "$V" /tmp/mtrl-gate-keys.txt --slot 2560 >/dev/null 2>&1; then
      echo "  [gate] FAIL: 无 p4view 二进制或构建失败（先 cargo build --release -p engramdb-bench --bin p4view）"
      exit 1
    fi
    echo "  [gate] verify 20k view ..."
    if ! target/release/p4view verify data/real-rows "$V" --keys /tmp/mtrl-gate-keys.txt --sub 100 >/dev/null 2>&1; then
      echo "  [gate] FAIL: 视图抽样校验不通过"
      exit 1
    fi
    RAW=$(target/release/p4view bench data/real-rows "$V" --keys probes/view-keys-20k.txt --threads 8 2>/dev/null)
    OUT=$(echo "$RAW" | grep -E "^B,|^A,")
    B8=$(echo "$OUT" | sed -n '2p')
    A8=$(echo "$OUT" | sed -n '3p')
    B8RPS=$(echo "$B8" | cut -d, -f3)
    A8RPS=$(echo "$A8" | cut -d, -f3)
    AMP=$(echo "$RAW" | grep "^amplification" | cut -d, -f5)
    echo "  [gate] B8t=$B8RPS rows/s  A8t=$A8RPS rows/s  ampl_B=$AMP"
    OK_B=$(awk -v b="$B8RPS" -v a="$A8RPS" 'BEGIN { print (b >= 2*a) ? 1 : 0 }')
    OK_A=$(awk -v amp="$AMP" 'BEGIN { print (amp <= 1.05) ? 1 : 0 }')
    if [[ "$OK_B" == "1" && "$OK_A" == "1" ]]; then
      echo "  [gate] PASS: 视图路径结构收益成立（B>=2xA, ampl<=1.05）"
    else
      echo "  [gate] FAIL: B/A 或放大不达标 B8=$OK_B A8=$OK_A"
      exit 1
    fi
  else
    echo "  [gate] 缺 probes/view-keys-20k.txt（先运行 p4view build 20000 ...）"
  fi
else
  echo "  [gate] 无真表数据：跳过基准（CI 同理）"
fi


echo "== [gate] decode baseline check =="
if [[ -f probes/cpu_tiny_baseline.csv && -f probes/qwen35_cpu_baseline.csv ]]; then
  if command -v python3 >/dev/null 2>&1; then
    if ! python3 scripts/decode_baseline_check.py >/tmp/decode_baseline_check.log 2>&1; then
      echo "  [gate] FAIL: decode baseline thresholds exceeded"
      cat /tmp/decode_baseline_check.log
      exit 1
    fi
    echo "  [gate] PASS: decode baseline thresholds satisfied"
  else
    echo "  [gate] python3 not found; skip"
  fi
else
  echo "  [gate] decode baseline CSVs missing; skip"
fi

echo "== [gate] PASS =="
