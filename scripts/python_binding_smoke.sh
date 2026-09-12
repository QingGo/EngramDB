#!/usr/bin/env bash
# Minimal Python bridge smoke: build the PyO3 extension (the one and only
# Python backend), then run the engram-peft disk-backed MultiHeadEmbedding
# self-check.
#
# The C ABI (`engramdb-cabi`) is built too, but only so `scripts/c_abi_smoke.py`
# can check that surface -- the Python package no longer loads it (roadmap §34).
set -euo pipefail
cd "$(dirname "$0")/.."

bash scripts/build_pyo3.sh
cargo build -p engramdb-cabi --release
PYTHONPATH=python python3 examples/interop_engram_peft.py
