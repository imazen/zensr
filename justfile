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
    cargo build --release --locked --manifest-path tools/coz-profile/Cargo.toml

coz-check:
    cargo fmt --manifest-path tools/coz-profile/Cargo.toml -- --check
    cargo clippy --manifest-path tools/coz-profile/Cargo.toml --all-targets --locked

coz-run profile +args:
    test ! -e "{{profile}}"
    test ! -e "{{profile}}.bin.sha256"
    sha256sum ./tools/coz-profile/target/release/zensr-coz-profile > "{{profile}}.bin.sha256"
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

# Sync imazen-26 image bytes from public R2 into the canonical checkout
# (manifests-only in git). Skips files already present. All three buckets:
# the effective split re-buckets ~180 files, so the train pool reads from all.
fetch-corpus:
    for b in train validate test; do nice -n 19 python3 tools/fetch_corpus_bytes.py $b; done

# Summarize paired raw/projected model gains with the image-level split.
canon4-calibration tsv:
    python3 tools/canon4_calibration.py "{{tsv}}"

# The environment must provide numpy and torch; the model directory also needs
# its local f32 training export, which is not checked in.
verify-adopted-f16 model:
    python3 tools/verify_adopted_f16.py "{{model}}"

# Reproduce calibration inputs without running model inference. Validate the
# identity scores against the original production TSV before joining results.
canon4-metadata root out:
    ZENSR_EVAL_METADATA=1 ZENSR_EVAL_JOBS=6 ZENSR_EVAL_QS=15,35,55,75,85,88,90,92,94,96 ZENSR_EVAL_ENCODERS=turbo,mozjpeg,jpegli,zenjpeg,zenjpeg:auto,zenjpeg:hybrid,zenjpeg:mozbase,zenjpeg:noaq ./target/release/dejpeg_eval "{{root}}" "{{out}}" 3 1 dejpeg_rt24i_canon4 dejpeg_rt24i_canon4

# Run inside the host's resource wrapper. Also exercises the committed model.
canon4-test:
    cargo test -p zensr-zenjpeg -p zensr-micro

canon4-lint:
    cargo fmt --all -- --check
    cargo clippy -p zensr-zenjpeg -p zensr-micro --all-targets -- -D warnings

canon4-fit *args:
    python3 tools/fit_canon4.py {{args}}

canon4-policy-check *args:
    python3 tools/check_canon4_policy.py {{args}}

# Empirical risk budgets are selected on calibration images, never audit images.
canon4-frontier *args:
    python3 tools/routing_frontier.py {{args}}

canon4-routing-tests:
    python3 -m unittest discover -s tools -p 'test_*canon4*.py'
    python3 -m unittest discover -s tools -p 'test_routing_frontier.py'

publication-routing-manifest *args:
    python3 tools/publication_routing_manifest.py {{args}}

publication-routing-inventory manifest:
    python3 -c 'import json,sys; from pathlib import Path; sys.path.insert(0,"tools"); from publication_routing_manifest import inventory; print(json.dumps(inventory(Path(sys.argv[1])),indent=2))' "{{manifest}}"

publication-routing-build:
    cargo build --release -p zensr-bench --bin routing_study --bin zjtool

publication-routing-score-tests:
    cargo test --locked -p zensr-bench --bin routing_study

publication-routing-score-check:
    cargo clippy --locked -p zensr-bench --bin routing_study -- -D warnings

publication-routing-cost-build:
    cargo build --release -p zensr-bench --features routing-cost --bin routing_cost

publication-routing-cost manifest *args:
    ZENSR_COST_MANIFEST="{{manifest}}" ./target/release/routing_cost {{args}}

publication-routing-cost-inputs *args:
    {{env_var_or_default("ZENSR_STUDY_PYTHON", "python3")}} tools/publication_routing_cost_inputs.py {{args}}

# Invoke under run-heavy. A system scope supplies a real cpuset boundary on
# hybrid CPUs; retain the same memory/no-swap cap inside that boundary.
publication-routing-cost-pinned manifest cpus pin *args:
    sudo systemd-run --scope --quiet -p AllowedCPUs="{{cpus}}" -p MemoryMax=16G -p MemorySwapMax=0 sudo -u "$(id -un)" env TMPDIR="{{env_var('TMPDIR')}}" RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ZENBENCH_PIN="{{pin}}" ZENBENCH_DATA_DIR="{{env_var('ZENBENCH_DATA_DIR')}}" ZENSR_COST_MAX_WALL_SECONDS="{{env_var_or_default('ZENSR_COST_MAX_WALL_SECONDS', '120')}}" ZENSR_COST_MANIFEST="{{manifest}}" nice -n19 ionice -c3 ./target/release/routing_cost --cpu-set="{{cpus}}" --cpu-policy=performance {{args}}

publication-routing-run manifest output role sizes qualities encoders jobs="6" exclusions="":
    ./target/release/routing_study "{{manifest}}" "{{output}}" "{{role}}" "{{sizes}}" "{{qualities}}" "{{encoders}}" "{{jobs}}" "{{exclusions}}"

[positional-arguments]
publication-routing-amend *args:
    python3 tools/publication_routing_amend.py "$@"

publication-routing-amend-tests:
    python3 -m unittest discover -s tools -p 'test_publication_routing_amend.py'

[positional-arguments]
publication-routing-scores-only *args:
    {{env_var_or_default("ZENSR_STUDY_PYTHON", "python3")}} tools/publication_routing_scores_only.py "$@"

publication-routing-scores-only-tests:
    {{env_var_or_default("ZENSR_STUDY_PYTHON", "python3")}} -m unittest discover -s tools -p 'test_publication_routing_scores_only.py'

[positional-arguments]
publication-routing-scatter *args:
    {{env_var_or_default("ZENSR_STUDY_PYTHON", "python3")}} tools/publication_routing_scatter.py "$@"

decoded-router-replay replay:
    ZENSR_ROUTER_REPLAY="{{replay}}" cargo test --locked -p zensr-zenjpeg decoded_router::tests::publication_replay -- --ignored --nocapture

decoded-router-jpeg-replay replay:
    ZENSR_ROUTER_JPEG_REPLAY="{{replay}}" cargo test --release --locked -p zensr-zenjpeg decoded_router::tests::publication_jpeg_features -- --ignored --nocapture

# Transport finalized cell tables beside retained artifacts for one local audit.
publication-routing-audit-bundle root output:
    tar --dereference -C "{{root}}" -I zstd -cf "{{output}}" cells IDENTITY.tsv FEATURES.tsv ENCODERS.tsv COVERAGE.tsv COMPLETE REFERENCE_EXCLUSION.json reference-exclusion-history

publication-routing-cells *args:
    {{env_var_or_default("ZENSR_STUDY_PYTHON", "python3")}} tools/publication_routing_cells.py {{args}}

publication-routing-retention *args:
    python3 tools/publication_routing_retention.py {{args}}

publication-routing-retention-tests:
    python3 -m unittest discover -s tools -p test_publication_retention.py

publication-routing-retention-inventory root:
    python3 tools/publication_routing_retention_inventory.py "{{root}}"

# Run beside storage only while the writer is frozen or ended.
# Even a storage-local directory scan can contend with active cell renames.
publication-routing-storage-count root:
    python3 -c 'import datetime,json,os,sys; n=sum(e.name.endswith(".tsv") for e in os.scandir(os.path.join(sys.argv[1],"cells"))); print(json.dumps(dict(checked=datetime.datetime.now(datetime.timezone.utc).isoformat(),completed_cells=n)))' "{{root}}"

publication-routing-leakage source output:
    {{env_var_or_default("ZENSR_STUDY_PYTHON", "python3")}} tools/publication_routing_leakage.py "{{source}}" "{{output}}"

publication-routing-analysis *args:
    {{env_var_or_default("ZENSR_STUDY_PYTHON", "python3")}} tools/publication_routing_analysis.py {{args}}

publication-routing-tests:
    {{env_var_or_default("ZENSR_STUDY_PYTHON", "python3")}} -m unittest discover -s tools -p 'test_publication_routing.py'

publication-routing-stage *args:
    {{env_var_or_default("ZENSR_STUDY_PYTHON", "python3")}} tools/publication_routing_stage.py {{args}}

publication-routing-remote-stage config logs *args:
    python3 tools/publication_routing_remote_stage.py "{{config}}" "{{logs}}" {{args}}

publication-routing-live config log guard:
    python3 tools/publication_routing_remote_stage.py "{{config}}" "{{log}}" --status --guard-pid "{{guard}}"

publication-routing-native-pilot *args:
    {{env_var_or_default("ZENSR_STUDY_PYTHON", "python3")}} tools/publication_routing_native_pilot.py {{args}}

publication-routing-stage-migrate *args:
    python3 tools/publication_routing_stage_migrate.py {{args}}

publication-routing-report source output:
    {{env_var_or_default("ZENSR_STUDY_PYTHON", "python3")}} tools/publication_routing_report.py "{{source}}" "{{output}}"

publication-routing-sizes *args:
    {{env_var_or_default("ZENSR_STUDY_PYTHON", "python3")}} tools/publication_routing_sizes.py {{args}}

publication-routing-lint:
    cargo fmt -p zensr-bench -- --check
    cargo clippy -p zensr-bench --bin routing_study --bin coef_features --bin dejpeg_eval -- -D warnings

publication-routing-storage-tests:
    cargo test -p zensr-bench --bin routing_study storage::tests -- --nocapture

publication-routing-storage-portable:
    mkdir -p "{{env_var('TMPDIR')}}"
    rustc --edition 2024 --test tools/test_routing_storage.rs -o "{{env_var('TMPDIR')}}/routing-storage-tests.exe"
    "{{env_var('TMPDIR')}}/routing-storage-tests.exe"

publication-routing-capture identity output:
    python3 tools/publication_routing_provenance.py "{{identity}}" "{{output}}"

publication-routing-resource-pilot source output edge *args:
    python3 tools/publication_routing_resource_pilot.py "{{source}}" "{{output}}" --edge "{{edge}}" {{args}}

publication-routing-audit-image output base:
    {{env_var_or_default("ZENSR_STUDY_PYTHON", "python3")}} tools/publication_audit_image.py "{{output}}" --base "{{base}}"

publication-routing-table-snapshot source output:
    {{env_var_or_default("ZENSR_STUDY_PYTHON", "python3")}} tools/publication_routing_tables.py "{{source}}" "{{output}}"

publication-routing-resume-check root *args:
    python3 tools/publication_routing_resume.py "{{root}}" {{args}}

# Replay existing artifacts through a frozen native transfer and full audit.
publication-routing-transfer-pilot config source archive reference logs:
    {{env_var_or_default("ZENSR_STUDY_PYTHON", "python3")}} -c 'import json,sys; from pathlib import Path; sys.path.insert(0,"tools"); from publication_routing_native_pilot import transfer_pilot; transfer_pilot(json.loads(Path(sys.argv[1]).read_text()),*[Path(p) for p in sys.argv[2:]])' "{{config}}" "{{source}}" "{{archive}}" "{{reference}}" "{{logs}}"

# Run beside storage; stop only the exact native-transfer Python child.
publication-routing-stop-native container:
    docker exec "{{container}}" python3 -c 'import os,signal,pathlib; rows=[(int(p.parent.name),p.read_bytes().split(b"\0")) for p in pathlib.Path("/proc").glob("[0-9]*/cmdline")]; ids=[pid for pid,args in rows if args[0]==b"python3" and b"/tools/publication_routing_native_stage.py" in args and b"drain" in args]; assert len(ids)==1,ids; os.kill(ids[0],signal.SIGTERM)'

# Retained inputs; no encoding or routing refit. Invoke under run-heavy.
projection-check manifest output:
    cargo run --release --locked -p zensr-bench --bin projection_check -- "{{manifest}}" "{{output}}"

projection-check-tests:
    cargo test --locked -p zensr-bench --bin projection_check

projection-check-inputs report root output count="4":
    python3 tools/projection_check_inputs.py "{{report}}" "{{root}}" "{{output}}" --count "{{count}}"

# New output directory; fits only on the immutable development table snapshot.
routing-feature-ablation root original output:
    {{env_var_or_default("ZENSR_STUDY_PYTHON", "python3")}} tools/routing_feature_ablation.py "{{root}}" "{{original}}" "{{output}}"

# Preserves the frozen model; explicitly measures its APP15-size sensitivity.
decoded-router-metadata-sensitivity replay output:
    ZENSR_ROUTER_JPEG_REPLAY="{{replay}}" ZENSR_METADATA_REPORT="{{output}}" cargo test --release --locked -p zensr-zenjpeg publication_metadata_sensitivity -- --ignored --nocapture

# Experimental selector is compiled only in tests; production stays fixed.
decoded-router-spatial-diagnostic manifest output:
    ZENSR_SPATIAL_INPUTS="{{manifest}}" ZENSR_SPATIAL_REPORT="{{output}}" cargo test --release --locked -p zensr-zenjpeg publication_spatial_windows -- --ignored --nocapture

# Secondary check only: this cohort was already inspected in the original study.
routing-feature-check development candidates reserved original_report output:
    {{env_var_or_default("ZENSR_STUDY_PYTHON", "python3")}} tools/routing_feature_check.py "{{development}}" "{{candidates}}" "{{reserved}}" "{{original_report}}" "{{output}}"

decoded-router-candidate-replay replay:
    ZENSR_CANDIDATE_REPLAY="{{replay}}" cargo test --release --locked -p zensr-zenjpeg candidate_publication_replay -- --ignored --nocapture

# Write raw RGB8; the example prints output dimensions.
repair-example input output:
    cargo run --release --locked -p zensr-zenjpeg --example repair -- "{{input}}" "{{output}}"

docs-check:
    python3 tools/check_docs.py

# Documentation maintenance; claim the checkout and start from a clean tree.
docs-move source destination:
    python3 tools/relocate_docs.py "{{source}}" "{{destination}}"

recover-pair-sources *args:
    {{env_var_or_default("ZENSR_PYTHON", "python3")}} tools/recover_pair_sources.py {{args}}

test-pair-recovery:
    {{env_var_or_default("ZENSR_PYTHON", "python3")}} -m unittest discover -s tools -p test_recover_pair_sources.py

verify-pair-sources *args:
    {{env_var_or_default("ZENSR_PYTHON", "python3")}} tools/verify_pair_sources.py {{args}}

pair-release *args:
    {{env_var_or_default("ZENSR_PYTHON", "python3")}} tools/pair_release.py {{args}}

test-pair-release:
    {{env_var_or_default("ZENSR_PYTHON", "python3")}} -m unittest discover -s tools -p test_pair_release.py

model-release *args:
    {{env_var_or_default("ZENSR_PYTHON", "python3")}} tools/model_release.py {{args}}

test-model-release:
    {{env_var_or_default("ZENSR_PYTHON", "python3")}} -m unittest discover -s tools -p test_model_release.py

# Weight-only canon4 QAT; use the workspace resource wrapper for training/evaluation.
qat-tests:
    {{env_var_or_default("ZENSR_PYTHON", "python3")}} -m unittest discover -s tools -p test_qat.py

qat-train *args:
    {{env_var_or_default("ZENSR_PYTHON", "python3")}} tools/qat/train.py {{args}}

qat-inputs *args:
    {{env_var_or_default("ZENSR_PYTHON", "python3")}} tools/qat/inputs.py {{args}}

qat-evaluator-build:
    cargo build --release --locked --manifest-path tools/qat/runtime/Cargo.toml

qat-evaluator-check:
    cargo clippy --locked --manifest-path tools/qat/runtime/Cargo.toml --all-targets -- -D warnings

qat-evaluator-tests:
    cargo test --release --locked --manifest-path tools/qat/runtime/Cargo.toml

qat-evaluator-debug-tests:
    cargo test --locked --manifest-path tools/qat/runtime/Cargo.toml

qat-evaluate *args:
    tools/qat/runtime/target/release/zensr-qat-eval {{args}}

qat-report *args:
    {{env_var_or_default("ZENSR_PYTHON", "python3")}} tools/qat/report.py {{args}}

qat-report-tests:
    {{env_var_or_default("ZENSR_PYTHON", "python3")}} -m unittest discover -s tools -p test_qat_report.py

# Requires a released CUDA device; never silently falls back to CPU.
qat-backend-check *args:
    {{env_var_or_default("ZENSR_PYTHON", "python3")}} tools/qat/check_backends.py {{args}}

qat-merge *args:
    {{env_var_or_default("ZENSR_PYTHON", "python3")}} tools/qat/merge.py {{args}}

qat-mirror *args:
    {{env_var_or_default("ZENSR_PYTHON", "python3")}} tools/qat/mirror.py {{args}}

qat-float-postquant *args:
    {{env_var_or_default("ZENSR_PYTHON", "python3")}} tools/qat/post_quantize.py {{args}}

qat-package *args:
    {{env_var_or_default("ZENSR_PYTHON", "python3")}} tools/qat/package.py {{args}}

# Quality-model quantization experiment; all generated bytes stay outside git.
quality-int8-build:
    cargo build --release --manifest-path tools/quality-int8/runtime/Cargo.toml

external-model-acquire root:
    python3 tools/external-models/acquire.py "{{root}}"

external-sr-acquire root model:
    python3 tools/external-models/acquire_sr.py "{{root}}" "{{model}}"

external-sr-evaluate *args:
    {{env_var_or_default("ZENSR_PYTHON", "python3")}} tools/external-models/evaluate_sr.py {{args}}

external-sr-verify *args:
    {{env_var_or_default("ZENSR_PYTHON", "python3")}} tools/external-models/verify_sr.py {{args}}

external-sr-verify-images *args:
    {{env_var_or_default("ZENSR_PYTHON", "python3")}} tools/external-models/verify_sr_images.py {{args}}

external-sr-sweep *args:
    python3 tools/external-models/sweep_sr.py {{args}}

external-sr-build:
    cargo build --release --locked --manifest-path tools/quality-int8/runtime/Cargo.toml --bin superres

external-model-summarize *args:
    python3 tools/external-models/summarize.py {{args}}

external-model-check:
    python3 -m unittest discover -s tools/external-models -p 'test_*.py'
    cargo fmt --manifest-path tools/quality-int8/runtime/Cargo.toml -- --check
    cargo clippy --locked --manifest-path tools/quality-int8/runtime/Cargo.toml --all-targets -- -D warnings

external-sr-cpu-build:
    cargo build --release --locked --manifest-path tools/quality-int8/runtime/Cargo.toml --bin cpu_sr

external-sr-cpu-worker *args:
    {{env_var_or_default("ZENSR_PYTHON", "python3")}} tools/external-models/cpu_worker.py {{args}}

external-sr-cpu-plan *args:
    python3 tools/external-models/cpu_plan.py {{args}}

external-dscf-export *args:
    {{env_var_or_default("ZENSR_PYTHON", "python3")}} tools/external-models/export_dscf.py {{args}}

external-dscf-cpu-report *args:
    python3 tools/external-models/cpu_dscf_report.py {{args}}

external-sr-cpu-report *args:
    python3 tools/external-models/cpu_report.py {{args}}

external-sr-cpu-progress *args:
    python3 tools/external-models/cpu_progress.py {{args}}

external-model-pilot *args:
    {{env_var_or_default("ZENSR_PYTHON", "python3")}} tools/external-models/pilot.py {{args}}

external-model-evaluate *args:
    {{env_var_or_default("ZENSR_PYTHON", "python3")}} tools/external-models/evaluate.py {{args}}

external-model-scorer-build:
    cargo build --release --locked --manifest-path tools/quality-int8/runtime/Cargo.toml --bin external

quality-int8-export *args:
    {{env_var_or_default("ZENSR_PYTHON", "python3")}} tools/quality-int8/export.py {{args}}

quality-int8-run *args:
    tools/quality-int8/runtime/target/release/zensr-quality-int8 {{args}}

quality-int8-bench *args:
    tools/quality-int8/runtime/target/release/latency {{args}}

quality-int8-check:
    cargo clippy --locked --manifest-path tools/quality-int8/runtime/Cargo.toml --all-targets -- -D warnings

quality-int8-reexport *args:
    {{env_var_or_default("ZENSR_PYTHON", "python3")}} tools/quality-int8/reexport.py {{args}}

quality-int8-scaling *args:
    ZENSR_THREAD_PARITY={{env_var_or_default("ZENSR_THREAD_PARITY", "each-process")}} tools/quality-int8/runtime/target/release/scaling {{args}}

quality-int8-report *args:
    {{env_var_or_default("ZENSR_PYTHON", "python3")}} tools/quality-int8/report.py {{args}}

quality-int8-stratify *args:
    {{env_var_or_default("ZENSR_PYTHON", "python3")}} tools/quality-int8/stratify.py {{args}}

quality-int8-combine *args:
    {{env_var_or_default("ZENSR_PYTHON", "python3")}} tools/quality-int8/combine.py {{args}}

quality-int8-report-tests:
    {{env_var_or_default("ZENSR_PYTHON", "python3")}} -m unittest discover -s tools -p test_quality_int8_report.py

quality-int8-latency-report *args:
    {{env_var_or_default("ZENSR_PYTHON", "python3")}} tools/quality-int8/latency_report.py {{args}}

quality-int8-package *args:
    {{env_var_or_default("ZENSR_PYTHON", "python3")}} tools/quality-int8/package.py {{args}}

quality-int8-rgb-grid exports output:
    tools/quality-int8/runtime/target/release/zensr-quality-int8 rgb-grid {{exports}} {{output}}

quality-int8-oracle *args:
    {{env_var_or_default("ZENSR_PYTHON", "python3")}} tools/quality-int8/oracle.py {{args}}

quality-int8-archive-remote *args:
    {{env_var_or_default("ZENSR_PYTHON", "python3")}} tools/quality-int8/archive_remote.py {{args}}
