# Browser super-resolution demo

`tools/browser-demo/` is a standalone local workbench for SPANF and DSCF.
It runs ONNX Runtime Web 1.31.1 inside a module worker, with selectable WebGPU
or a pool of single-threaded WASM tile workers. Uploaded JPEG/PNG pixels stay in the browser;
the server serves static assets and has no upload endpoint. This is an
experimental browser integration, separate from the native Rust runtime.

The UI accepts files up to 64 MiB and 12.6 million pixels, with each side at most
8192 pixels. Both models infer at ×4. Selecting ×2 applies an encoded-RGB box
resize to floating-point samples before their single final RGB8 conversion.
Browser decoding and an sRGB canvas provide input pixels. Transparency is
composited on white; HDR, original metadata and 16-bit output are not preserved.
JPEG artifact repair is not implemented in this browser demo.

## Memory behavior

Inference uses 128×128 input cores with clamped halos: 16 input
pixels for SPANF, 21 for DSCF. At true image boundaries the model receives the
actual edge shape. Padding raw input to a fixed shape would change hidden-layer
boundary behavior. Completed RGB8 cores enter one horizontal band, which the
shared zenpng WASM encoder consumes incrementally. Chunk acknowledgments provide
backpressure. A preview is limited to 1400 pixels on its longest side.

Auto CPU concurrency follows the reported logical CPU count, available tile
work and an explicit aggregate inference budget. Each CPU worker has a 64 MiB
WASM maximum and an 80 MiB reservation including transferred tile buffers.
The default budget uses one quarter of the coarse `deviceMemory` report, capped
at 2 GiB, or 512 MiB when that API is unavailable. It is not a free-RAM reading.
A bounded raster-order queue keeps at most one outstanding result per worker;
completed cores still enter a single output band. WebGPU remains one tile at a
time, with a 256 MiB WASM support-memory cap. Each codec module has a
128 MiB linker maximum. WebGPU allocation instrumentation caps the aggregate
live sizes of buffers on the runtime's actual GPU device at 256 MiB, including
buffers retained in its pool. Exceeding a cap fails the job explicitly. The
browser's decoder, canvas backing stores, JS garbage collector, GPU driver and
browser process are outside these caps. This is **not a hard limit on tab RSS**.

Completed PNG bytes stream to the origin-private file system (OPFS) on a secure
origin. Download the PNG to keep a durable copy: browser storage may be evicted.
Saved results remain available after page reload; cancellation preserves them.
Without OPFS, the Blob fallback allows at most 128 MiB combined across retained
results and the active output. Reloading releases those in-memory results.
A 12 MP ×4 output can exceed this fallback, so use HTTPS with OPFS.

## Build and serve

Required tools: Node/npm, Rust with `wasm32-unknown-unknown`, wasm-bindgen CLI
0.2.129, and Python with ONNX and NumPy. The shared codec dependencies use pinned
Git revisions because this demo needs their retained streaming/probe APIs.
The PNG/resampling and JPEG-probe modules are separate to isolate their differing
zencodec/zenpixels dependency versions. zenflate uses opt-level 1 for this WASM
build; the initial opt-level 3 compilation crashed rustc on the tested toolchain.

```sh
just browser-deps
just browser-build
ZENSR_PYTHON=/path/to/python just browser-prepare /path/to/sr-cpu-run /path/to/dscf-cpu-run
just browser-serve --assets tools/browser-demo/dist --gallery /path/to/sr-cascade-run \
  --port 3331 --cert /path/to/cert.pem --key /path/to/key.pem
```

Use a locally trusted certificate for access from another computer. WebGPU and
OPFS require a secure context. The read-only server binds on all interfaces and
sets cross-origin isolation headers. Keep certificates outside the repository.
`dist/`, npm dependencies, built WASM and model binaries are ignored by Git.

`prepare.py` checks SPANF's retained FP32 initializer bytes and mixed stored
precision bytes before expanding the same f16 weights/f32 biases for ONNX FP32
compute. DSCF keeps the retained author FP32 weights. Only spatial shape
declarations change; operation bytes must match the source exports. The browser
verifies model SHA-256 hashes before creating a session. Model source and license
details remain in [the external-model evaluation](research/EXTERNAL_MODELS_2026-11-09.md).

## Verification

The unit, codec, browser, parity and lifecycle recipes are in the root justfile.
Set `ZENSR_CHROME` to the browser executable for browser tests, and run heavy
tests through the workspace resource wrapper. `browser-test` takes input path,
model, scale, backend and output directory. `browser-parity` compares full-frame
and tiled WASM inference, including true edges. `browser-lifecycle` takes a
small JPEG, a large input and a JSON report path, checking cancellation, restart,
saved results after reload and malformed input rejection.

Browser backend results are not established as numerically identical to native
CPU or CUDA. The functional tests are not a perceptual quality study or a
hardware WebGPU benchmark.
