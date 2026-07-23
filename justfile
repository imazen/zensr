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
