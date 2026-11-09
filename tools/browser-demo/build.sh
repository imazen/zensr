#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
: "${TMPDIR:=$HOME/tmp/zensr-browser}"
export TMPDIR
mkdir -p "$TMPDIR"
export RUSTFLAGS='-C target-feature=+simd128 -C link-arg=--max-memory=134217728'
cargo build --locked --release --target wasm32-unknown-unknown --manifest-path codec/Cargo.toml
wasm-bindgen codec/target/wasm32-unknown-unknown/release/zensr_browser_codec.wasm --target web --out-dir codec/pkg
for mode in relaxed unfused reference; do
  features=+simd128
  args=()
  if [ "$mode" = relaxed ]; then features=+simd128,+relaxed-simd; fi
  if [ "$mode" = reference ]; then args=(--features strict-fma); folder=pkg; else folder=pkg-$mode; fi
  export RUSTFLAGS="-C target-feature=$features -C link-arg=--max-memory=1073741824"
  cargo build --locked --release --target wasm32-unknown-unknown --manifest-path repair/Cargo.toml "${args[@]}"
  wasm-bindgen repair/target/wasm32-unknown-unknown/release/zensr_browser_repair.wasm --target web --out-dir repair/$folder
done
