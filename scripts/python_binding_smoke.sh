
#!/usr/bin/env bash
# Minimal Python bridge smoke: build C ABI cdylib, then run the engram-peft
# disk-backed MultiHeadEmbedding self-check.
set -euo pipefail
cd "$(dirname "$0")/.."

cargo build -p engramdb-python --release
PYTHONPATH=python python3 examples/interop_engram_peft.py
