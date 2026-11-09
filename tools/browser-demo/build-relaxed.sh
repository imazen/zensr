#!/usr/bin/env bash
# Build the unchanged native codec + zensr sources for the selected WASM capability.
set -euo pipefail
output=$(realpath -m "${1:?provide a new artifact directory}")
mode=${2:-relaxed}
case "$mode" in
  relaxed) features=+simd128,+relaxed-simd ;;
  unfused) features=+simd128 ;;
  *) echo "Expected relaxed or unfused arithmetic" >&2; exit 2 ;;
esac
: "${TMPDIR:=$HOME/tmp/zensr-browser}"
export TMPDIR
mkdir -p "$TMPDIR"
mkdir "$output"
export RUSTFLAGS="-C target-feature=$features -C link-arg=--max-memory=1073741824"
export CARGO_TARGET_DIR="${ZENSR_BROWSER_TARGET_DIR:-$output/target}"
cargo build --locked --release --target wasm32-unknown-unknown --manifest-path "$(dirname "$0")/repair/Cargo.toml"
wasm-bindgen "$CARGO_TARGET_DIR/wasm32-unknown-unknown/release/zensr_browser_repair.wasm" --target web --out-dir "$output/pkg"
python3 - "$mode" "$features" "$output" <<'IDENTITY'
import hashlib,json,subprocess,sys
from pathlib import Path
mode,features,directory=sys.argv[1:]
root=Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
output=Path(directory)
report=dict(mode=mode,target_features=features,source_patch=False,
    commit=subprocess.check_output(['jj','log','-r','@','--no-graph','-T','commit_id'],text=True).strip(),
    package={p.name:hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in (output/'pkg').iterdir() if p.is_file()})
(output/'BUILD_IDENTITY.json').write_text(json.dumps(report,indent=2)+'\n')
IDENTITY
