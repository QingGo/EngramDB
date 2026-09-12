#!/usr/bin/env bash
# Build the PyO3 native extension and copy it into the Python package.
# Uses a separate CARGO_HOME if the default one is not writable in this sandbox.
set -euo pipefail
cd "$(dirname "$0")/.."

export CARGO_HOME="${CARGO_HOME:-/tmp/cargo-home}"
case "$(uname -s)" in
  Darwin)
    export RUSTFLAGS="${RUSTFLAGS:-} -C link-arg=-undefined -C link-arg=dynamic_lookup"
    SRC=target/release/lib_engramdb.dylib
    ;;
  Linux)  SRC=target/release/lib_engramdb.so ;;
  MINGW*|MSYS*|CYGWIN*) SRC=target/release/engramdb.dll ;;
  # 曾经这里硬编码 `lib_engramdb.dylib`，导致本脚本在 Linux 上必然失败（cp 报错退出）。
  *) echo "build_pyo3.sh: unsupported platform $(uname -s)" >&2; exit 1 ;;
esac

cargo build -p engramdb-pyo3 --release
cp "$SRC" python/engramdb/_engramdb.so
echo "built python/engramdb/_engramdb.so from $SRC"
