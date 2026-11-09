#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
: "${TMPDIR:=$HOME/tmp/zensr-browser}"
export TMPDIR
mkdir -p "$TMPDIR"
export RUSTFLAGS='-C target-feature=+simd128 -C link-arg=--max-memory=134217728'
for crate in codec jpeg-probe; do
  cargo build --locked --release --target wasm32-unknown-unknown --manifest-path "$crate/Cargo.toml"
  if [ "$crate" = codec ]; then stem=zensr_browser_codec; else stem=zensr_browser_jpeg_probe; fi
  wasm-bindgen "$crate/target/wasm32-unknown-unknown/release/$stem.wasm" --target web --out-dir "$crate/pkg"
done

export RUSTFLAGS='-C target-feature=+simd128 -C link-arg=--max-memory=1073741824'
cargo build --locked --release --target wasm32-unknown-unknown --manifest-path repair/Cargo.toml
wasm-bindgen repair/target/wasm32-unknown-unknown/release/zensr_browser_repair.wasm --target web --out-dir repair/pkg
