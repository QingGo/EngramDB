
#!/usr/bin/env bash
# Build the engramdb-python mixed Rust/PyO3 wheel with maturin.
# Usage: MATURIN=<path-to-maturin> PYTHON=<python> bash scripts/build_wheel.sh
set -euo pipefail
cd "$(dirname "$0")/.."

MATURIN="${MATURIN:-maturin}"
PYTHON="${PYTHON:-python3}"
export CARGO_HOME="${CARGO_HOME:-/tmp/cargo-home}"
if [[ "$(uname -s)" == "Darwin" ]]; then
  export RUSTFLAGS="${RUSTFLAGS:-} -C link-arg=-undefined -C link-arg=dynamic_lookup"
fi

cd python
"$MATURIN" build --release --interpreter "$PYTHON"
echo "wheel written under ../target/wheels/"
