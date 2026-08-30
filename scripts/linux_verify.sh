#!/usr/bin/env bash
# Verify the published engramdb-python wheel on a real Linux host (WSL or Pi).
# Usage: bash scripts/linux_verify.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3}"
VERSION="-e"

"$PYTHON" -m pip install --upgrade "engramdb-python==${VERSION}"
"$PYTHON" scripts/python_wheel_smoke.py
