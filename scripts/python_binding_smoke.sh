#!/usr/bin/env bash
# Minimal Python bridge smoke: build the PyO3 extension (plus C-ABI fallback),
# then run the engram-peft disk-backed MultiHeadEmbedding self-check.
set -euo pipefail
cd "$(dirname "$0")/.."

bash scripts/build_pyo3.sh
cargo build -p engramdb-python --release
PYTHONPATH=python python3 examples/interop_engram_peft.py
