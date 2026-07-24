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
    printf '[patch.crates-io]\nzenjpeg = { path = "/home/lilith/work/zen/zenjpeg/zenjpeg" }\n' > .cargo/config.toml
