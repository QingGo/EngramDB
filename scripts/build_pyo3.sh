
#!/usr/bin/env bash
# Build the PyO3 native extension and copy it into the Python package.
# Uses a separate CARGO_HOME if the default one is not writable in this sandbox.
set -euo pipefail
cd "$(dirname "$0")/.."

export CARGO_HOME="${CARGO_HOME:-/tmp/cargo-home}"
if [[ "$(uname -s)" == "Darwin" ]]; then
  export RUSTFLAGS="${RUSTFLAGS:-} -C link-arg=-undefined -C link-arg=dynamic_lookup"
fi

cargo build -p engramdb-pyo3 --release
cp target/release/lib_engramdb.dylib python/engramdb/_engramdb.so
echo "built python/engramdb/_engramdb.so"
