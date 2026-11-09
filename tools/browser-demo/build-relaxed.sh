#!/usr/bin/env bash
# Run through run-heavy. Build an isolated arithmetic experiment, preserving native crates.
set -euo pipefail
output=$(realpath -m "${1:?provide a new artifact directory}")
: "${TMPDIR:=$HOME/tmp/zensr-browser}"
export TMPDIR
mkdir -p "$TMPDIR"
mode=${2:-relaxed}
case "$mode" in
  relaxed) features=+simd128,+relaxed-simd ;;
  unfused) features=+simd128 ;;
  *) echo "Expected relaxed or unfused arithmetic" >&2; exit 2 ;;
esac
python3 "$(dirname "$0")/prepare_relaxed.py" "$output/source" --mode "$mode"
export CARGO_TARGET_DIR="$output/target"
export RUSTFLAGS="-C target-feature=$features -C link-arg=--max-memory=1073741824"
cargo build --locked --release --target wasm32-unknown-unknown --manifest-path "$output/source/repair/Cargo.toml"
wasm-bindgen "$output/target/wasm32-unknown-unknown/release/zensr_browser_repair.wasm" --target web --out-dir "$output/pkg"
