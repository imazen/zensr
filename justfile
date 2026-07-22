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
