export:
    python3 tools/export_ntire25.py 128x128 256x256 512x512

dump:
    python3 tools/dump_spanf_weights.py

bench:
    cargo build --release -p zensr-bench
    ./target/release/zensr-bench models/SPANF_x4_128x128.onnx 128 128 10

verify:
    cargo build --profile release-fast -p zensr-micro
    ./target/release-fast/zensr-verify models

size:
    cargo build --profile release-min -p zensr-micro-abi
    ls -l target/release-min/libzensr_micro_abi.so

dump-adopted:
    python3 tools/dump_adopted.py

adopted-verify:
    cargo build --profile release-fast -p zensr-micro
    ./target/release-fast/zensr-adopted-verify models/adopted

systems-eval:
    cargo build --profile release-fast -p zensr-bench --bin systems_eval
    ./target/release-fast/systems_eval /mnt/v/imazen-26 benchmarks/systems_eval_$(date -u +%Y-%m-%d).tsv 8 12

distill-data:
    ~/work/zen/scripts/run-heavy -- python3 tools/make_distill_data.py

distill-train steps="60000":
    python3 tools/train_distill.py {{steps}}

ert-eval:
    cargo build --profile release-fast -p zensr-bench --bin ert_eval
    ./target/release-fast/ert_eval /mnt/v/imazen-26 benchmarks/systems_eval_$(date -u +%Y-%m-%d)_ert.tsv 8 12

summarize tsv:
    ./target/release-fast/systems_eval summarize {{tsv}}

systems-bench:
    cargo build --profile release-fast -p zensr-bench --bin systems_bench
    ./target/release-fast/systems_bench 5

people-pull n_shards="40" quota="2500":
    nice -n 19 ionice -c 3 python3 tools/pxhere_people_pull.py {{n_shards}} {{quota}}

people-eval:
    cargo build --profile release-fast -p zensr-bench --bin systems_eval
    ./target/release-fast/systems_eval /mnt/v/input/zensr-people-eval-root benchmarks/people_eval_$(date -u +%Y-%m-%d).tsv 64 12

people-gt-data:
    ZENSR_PEOPLE_OUT=~/tmp/zensr-people-v1 ~/work/zen/scripts/run-heavy -- python3 tools/make_people_gt_data.py 24000

unsplash-pull target="300":
    nice -n 19 python3 tools/unsplash_people_pull.py {{target}}

report:
    cargo build --profile release-fast -p zensr-bench --bin gallery_dump
    ./target/release-fast/gallery_dump ~/tmp/zensr-gallery/scenes.tsv ~/tmp/zensr-gallery 12
    python3 tools/build_report.py

cargo-local:
    mkdir -p .cargo
    printf '# LOCAL zen dep overrides (gitignored). Relative: works on every box.\n# Regenerate: just cargo-local\n[patch."https://github.com/imazen/zenjpeg"]\nzenjpeg = { path = "../zenjpeg/zenjpeg" }\n' > .cargo/config.toml

# Regenerate the effective imazen-26 split. Read by BOTH the Python trainer and
# the Rust eval harness — they must agree, and the near-duplicate same-bucketing
# moves 180 files across the held-out boundary, so the raw canonical buckets are
# not a safe fallback. Needs the canonical corpus (github.com/imazen/imazen-26,
# by default ~/work/zen/imazen-26; override with IMAZEN26_REPO).
split:
    python3 tools/corpus_split.py --write eval_split/imazen26_effective_split.tsv

# Is the local canonical-corpus checkout actually the published corpus? Verifying
# against the manifest is NOT the same question: 5314's manifest sha256 is the
# hash of its corrupt bytes, so a manifest check passes on a file that is 54% NUL
# and will not decode (imazen/imazen-26#2). This checks R2 instead, plus a local
# decode + zeroed-block scan. Run it after any corpus re-sync.
verify-corpus *ARGS:
    python3 tools/verify_canonical_corpus.py {{ARGS}}

# Rebuild the XL eval corpus, then prove it holds no training content.
xl-corpus OUT="/mnt/v/zensr/xl-eval-corpus":
    bash tools/build_xl_corpus.sh {{OUT}}
    find -L {{OUT}} -type f \( -iname '*.png' -o -iname '*.jpg' \) | sort > ~/tmp/xl-files.txt
    python3 tools/leakage_audit.py --files ~/tmp/xl-files.txt

# What the split does, and where the near-duplicate groups are.
split-report:
    python3 tools/corpus_split.py

# Standalone causal profiler: does not resolve the benchmark codec graph.
coz-build:
    cargo build --release --manifest-path tools/coz-profile/Cargo.toml

coz-check:
    cargo fmt --manifest-path tools/coz-profile/Cargo.toml -- --check
    cargo clippy --manifest-path tools/coz-profile/Cargo.toml --all-targets --locked

coz-run profile +args:
    test ! -e "{{profile}}"
    {{env_var_or_default("COZ_BIN", "coz")}} run --source-scope '%/zensr-micro/src/%' --output "{{profile}}" --- ./tools/coz-profile/target/release/zensr-coz-profile {{args}}


# Repeat the deterministic realtime probe and compare every output byte.
coz-repro outdir:
    mkdir -p "{{outdir}}"
    ./tools/coz-profile/target/release/zensr-coz-profile models/adopted/dejpeg_rt24g/weights_f16.raw 24 8 1 64 64 1 0 "{{outdir}}/baseline-a.f32"
    ./tools/coz-profile/target/release/zensr-coz-profile models/adopted/dejpeg_rt24g/weights_f16.raw 24 8 1 64 64 1 0 "{{outdir}}/baseline-b.f32"
    cmp "{{outdir}}/baseline-a.f32" "{{outdir}}/baseline-b.f32"
    sha256sum "{{outdir}}/baseline-a.f32" "{{outdir}}/baseline-b.f32"

# Compile and run the runtime's actual unit-test source in the isolated graph.
coz-test *args:
    cargo test --manifest-path tools/coz-profile/Cargo.toml --test runtime_unit {{args}}

# Exactness: raw model output must not depend on threads, tile size or tier.
exact-matrix out:
    cargo build --release --manifest-path tools/coz-profile/Cargo.toml
    python3 tools/exactness/matrix.py --bin tools/coz-profile/target/release/zensr-coz-profile --out "{{out}}"

# Exactness of final RGB8 through restore_jpeg on the recorded counterexamples.
exact-product out:
    cargo build --release --manifest-path tools/exactness/product/Cargo.toml
    python3 tools/exactness/product.py --bin tools/exactness/product/target/release/zensr-exactness-product --out "{{out}}" --jpeg /mnt/v/output/zensr/exactness-2026-10-18/inputs/girl-576x576-q40-420.jpg --jpeg /mnt/v/output/zensr/exactness-2026-10-18/inputs/city-576x576-q40-420.jpg --model q7=models/adopted/dejpeg7_graphics/weights_f16.raw:64:16 --model rt24g=models/adopted/dejpeg_rt24g/weights_f16.raw:24:8 --scalar-threads 16 --expect tools/exactness/product-expect-2026-10-18.json
