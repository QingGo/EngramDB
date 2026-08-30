#!/usr/bin/env bash
# Prepare the real Qwen3.5-0.8B model for EngramDB A/B benchmarks.
#
# This script never commits model data.  It only creates/refreshes the
# git-ignored data symlink and optionally copies/verifies the model into a local
# WSL-accessible directory.
#
# Examples:
#
#   # Create the usual macOS symlink from the external SSD.
#   scripts/prep_real_model.sh
#
#   # Symlink from a custom source.
#   scripts/prep_real_model.sh --source /path/to/Qwen3.5-0.8B
#
#   # Copy into a WSL-visible directory and verify all required files.
#   scripts/prep_real_model.sh --action copy \
#       --copy-to /mnt/c/Users/me/engramdb-transfer/Qwen3.5-0.8B
#
#   # Verify an existing local copy without changing anything.
#   scripts/prep_real_model.sh --action verify --source /path/to/Qwen3.5-0.8B
set -euo pipefail
cd "$(dirname "$0")/.."

SOURCE="${SOURCE:-/Volumes/My Passport/model/Qwen3.5-0.8B}"
DEST="${DEST:-data/Qwen3.5-0.8B}"
ACTION="${ACTION:-symlink}"
COPY_TO="${COPY_TO:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source) SOURCE="$2"; shift 2 ;;
    --dest) DEST="$2"; shift 2 ;;
    --action) ACTION="$2"; shift 2 ;;
    --copy-to) COPY_TO="$2"; shift 2 ;;
    -h|--help)
      grep '^#' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

REQUIRED=(
  "config.json"
  "configuration.json"
  "model.safetensors.index.json"
  "model.safetensors-00001-of-00001.safetensors"
  "tokenizer.json"
  "tokenizer_config.json"
)

if [[ ! -d "$SOURCE" ]]; then
  echo "ERROR: source model directory not found: $SOURCE" >&2
  exit 1
fi

verify_model() {
  local dir="$1"
  local missing=0
  for f in "${REQUIRED[@]}"; do
    if [[ ! -s "$dir/$f" ]]; then
      echo "  MISSING: $dir/$f" >&2
      missing=1
    fi
  done
  if [[ "$missing" -ne 0 ]]; then
    echo "ERROR: model verification failed in $dir" >&2
    exit 1
  fi
  echo "OK: required files present in $dir"
}

case "$ACTION" in
  symlink)
    mkdir -p "$(dirname "$DEST")"
    if [[ -e "$DEST" && ! -L "$DEST" ]]; then
      echo "ERROR: $DEST exists and is not a symlink; remove it first" >&2
      exit 1
    fi
    ln -sfn "$SOURCE" "$DEST"
    echo "symlink: $DEST -> $SOURCE"
    verify_model "$DEST"
    ;;
  copy)
    if [[ -z "$COPY_TO" ]]; then
      echo "ERROR: --action copy requires --copy-to" >&2
      exit 1
    fi
    mkdir -p "$COPY_TO"
    if command -v rsync >/dev/null 2>&1; then
      rsync -a --delete "$SOURCE/" "$COPY_TO/"
    else
      cp -a "$SOURCE/." "$COPY_TO/"
    fi
    echo "copied: $SOURCE -> $COPY_TO"
    verify_model "$COPY_TO"
    ;;
  verify)
    verify_model "$SOURCE"
    ;;
  *)
    echo "ERROR: unknown --action '$ACTION' (expected symlink|copy|verify)" >&2
    exit 2
    ;;
esac

echo "done."
