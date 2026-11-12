# Browser SR against author PyTorch — 2026-11-09

The downloaded browser pixels closely reproduce full-frame author PyTorch on this bounded four-input check. They are not bit-identical. A separate display regression reduced the SR overview to input resolution; fixing it preserves the downloaded PNG bytes. This is an implementation comparison, **not a score against an HR reference or evidence of perceptual model quality**.

## Inputs and controls

Four fixed inputs: 259×263 top-left crops of retained photo 1017, plot 7109 and web screenshot 8157, plus a 17×13 RGB pattern. The odd crops span nine 128-pixel cores and include actual image borders. Selection used source identity and geometry before scores. Input PNGs use zenpng; Pillow readback verifies their exact RGB8 bytes. Repair is off. Both ×4 and ×2 run through the actual browser UI, tiled workers, output conversion and streamed PNG download. WASM uses two tile workers. WebGPU uses SwiftShader, **not a physical GPU**.

PyTorch uses the pinned author modules and checkpoints, strict state loading, CPU FP32 and four threads. Only hard-coded CUDA device selections in constructors are redirected to CPU; forward code is unchanged. SPANF has two controls: original author FP32 and the browser's f16-stored weights expanded to FP32 (biases remain FP32). DSCF retains author FP32 weights. ×2 controls average aligned 2×2 cells of the full ×4 float result before the final RGB8 conversion.

SSIMULACRA2 below uses `fast-ssim2 =0.9.0`, score-only RGB8 sRGB. Retained reports also include the historical 0.8.2 score separately; older study scores have not been rewritten. Identical-byte control scores 100. Sparse one-level differences can still produce substantial score changes on these graphics, so channel counts and maximum differences accompany the scores. No IQA maps are requested or saved.

## ×4 versus original author FP32

| Input | SPANF WASM | SPANF WebGPU | DSCF WASM | DSCF WebGPU |
|---|---:|---:|---:|---:|
| Photo | 97.0849 | 97.1070 | 98.0075 | 98.1083 |
| Plot | 90.1444 | 90.1938 | 93.4214 | 93.4880 |
| Web graphics | 93.6998 | 93.6555 | 94.5347 | 94.6574 |
| Tiny pattern | 98.1193 | 98.1248 | 99.5551 | 99.5129 |

## ×2 versus original author FP32

| Input | SPANF WASM | SPANF WebGPU | DSCF WASM | DSCF WebGPU |
|---|---:|---:|---:|---:|
| Photo | 97.8381 | 97.8249 | 97.8943 | 97.8885 |
| Plot | 92.2806 | 92.2177 | 92.7596 | 92.7480 |
| Web graphics | 95.3400 | 95.2749 | 95.4676 | 95.2844 |
| Tiny pattern | 98.5110 | 98.5110 | 98.8441 | 98.8462 |

## ×4 with matching stored weights

| Input | SPANF WASM / WebGPU | DSCF WASM / WebGPU |
|---|---:|---:|
| Photo | 98.1926 / 98.1300 | 98.0075 / 98.1083 |
| Plot | 93.6761 / 93.4695 | 93.4214 / 93.4880 |
| Web graphics | 94.8179 / 94.8061 | 94.5347 / 94.6574 |
| Tiny pattern | 99.7534 / 100.0000 | 99.5551 / 99.5129 |

At ×4, matching-weight WASM SPANF changes 62, 28, 73 and 1 channel values respectively; DSCF changes 75, 32, 75 and 4. Each large output has 3,269,616 values; the tiny output has 10,608. Every matching-weight difference is at most 1/255 in both backends. Against original FP32 SPANF, the maximum is 2/255. ×2 maximum differences are also 2/255 for SPANF and 1/255 for DSCF. These observations do not establish exact cross-runtime parity or cover arbitrary images.

## Display correction and validation

Before the fix, the 259×263 input produced a correct 1036×1052 downloaded PNG but only a 259×263 overview. The regression fails on the old code and passes with an overview based on output dimensions. The overview retains the existing 4 MP / 4096-side budget; native inspection tiles are unchanged. A repeated run checks byte-identical downloaded PNG output.

The shared native zenjpeg integration also replays all 136 frozen JPEG cases with exact legacy RGB8 and routing (60 restored). Browser JPEG tests cover all eight EXIF orientations, identity sRGB ICC, malformed ICC errors, relaxed-first loading and forced unfused fallback. Arbitrary ICC profiles and all native row/output paths are not established by these browser checks.

## Reproduction and retained evidence

Recipes: `browser-torch-reference`, `browser-torch-test`, `browser-torch-score`, `browser-preview-test`, `browser-sr-diff-build`. The scripts require explicit artifact/model paths. Author sources: [SPANF](https://github.com/Amazingren/NTIRE2025_ESR), [DSCF](https://github.com/Yaozzz666/DSCF-SR). Model/source hashes, exact commands, raw outputs, lossless browser PNGs and per-case scores live under `/mnt/v/output/zensr/browser-torch-20261109/`, with companion execution logs in `browser-repair-20261109/`. No generated images or model binaries are added to Git.

Measured on i265: author control run 4 s, run-heavy peak RSS 1.07 GiB; WASM browser batch 13 s / 0.18 GiB; SwiftShader batch 176 s / 0.19 GiB. These wrapper RSS figures describe the monitored command and are **not a whole-browser process-tree memory bound**. Timing includes orchestration and is not a CPU-versus-GPU performance benchmark.
