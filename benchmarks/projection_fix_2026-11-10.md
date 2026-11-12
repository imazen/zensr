# Partial-block projection correction — 2026-11-10

The shared runtime now reconstructs hidden samples from a partial block's
stored coefficients instead of replicating candidate edge pixels. Hidden
IDCT values remain unclipped. The coefficient limits, guards and model weights
are unchanged; fully visible blocks keep their existing arithmetic. The fix
uses the existing fixed-size block scratch, without a full-frame padded image.
Source: zensr `0bb29cf63a4ab8fa8864994acc659e61c890c5e6`.

The native decoder pins the correction through zenjpeg
`06d0e6f468285cd9fccbf7abae195ef31a6bc4fa`; relaxed, unfused and strict browser
builds use that same dependency chain. Review PRs remain unmerged.

## Delivered pixels and quality

All **16,104 JPEG cases / 244 origins** from the frozen small bank replay the
previous projection hashes exactly through an explicitly padded reference.
All **4,928 fully visible cases** remain RGB8-identical under the corrected
runtime. 7,204 partial cases change. Every corrected image is retained as IMI.
Sources remain 161 train / 83 validate, LR long edges 32/64/128, whole frames,
q5–100, 420/444. Original JPEGs, raw model output and all old results remain intact.

| Split | Cases | Mean ΔA | Mean ΔSSIM2 | Loss >0.5 SSIM2 versus old | Worst ΔSSIM2 |
|---|---:|---:|---:|---:|---:|
| train | 10,626 | +0.03005 | +0.01276 | 22 | -2.41581 |
| validate | 5,478 | +0.09381 | +0.00650 | 11 | -1.04593 |
| combined | 16,104 | +0.05174 | +0.01063 | 33 | -2.41581 |

These are forced repair results, not a router operating point. Under the existing
quality-dependent harm curve relative to plain decode, forced harmed cases
change from 2,210 to 2,146: 78 resolved and 14 newly harmed. No curve, threshold,
skip or model-selection criterion was relaxed. No per-image quality guarantee
follows. The worst old-to-new SSIM2 decline is 9554/e32/q20/420 (AI product image):
76.6983 → 74.2825, still above its plain-decode score 69.2262.
The worst A decline is -7.1495 at 6814/e32/q5/420. Full category × Q × sampling ×
size tables, per-image tails and both split summaries are retained.

## Former worst SR cases

On the original tiny JPEGs, guarded-repair max RGB spreads 38/43 previously
became 92/94 after constraints. Corrected constraints retain 38/43. A new GPU
witness run reruns all 69 source-3313 cases with unchanged inputs/models;
all no-repair SR pixels and scores replay the original exactly.

| Original failure | Previous repair-first A | Corrected repair-first A | No-repair SR A | Corrected versus no repair |
|---|---:|---:|---:|---:|
| SPANF, e32/q40/420 | -67.3436 | -16.5333 | -20.9195 | +4.3862 |
| DSCF, e32/q50/420 | -58.0921 | -10.6058 | -13.4008 | +2.7950 |

The gallery now provides previous-versus-corrected SR flicker links, HR/direct
controls and native-LR stages. The old JPEG tint is intentionally preserved to
isolate the constraint fix. The separate encoder-padding correction is in the
same native review PR. Small learned SR chroma and its zero-padding behavior
are unchanged.

## Verification

- A decoded partial-block invariance regression fails before the fix and passes
  after it for both coefficient orders, all 1–8 pixel widths/heights and three
  quantizer steps. Additional tests cover unclipped hidden IDCT values and
  error to the decoded target in mixed partial geometries.
- Existing product library tests and strict lint/format checks pass. Existing
  explicit publication tests remain caller-selected; no expectations changed.
- All 136 native integration cases match the compatibility API's corrected
  RGB8 and routing, with 60 restored. Original route decisions and geometries
  remain unchanged in all 136 cases.
- Strict WASM/native RGB8 parity, identity sRGB ICC, malformed ICC rejection
  and all eight EXIF orientations pass. Real browser tests pass relaxed-first,
  injected relaxed failure selecting unfused, total-init failure, Blob fallback
  and independent 1× repair. Viewer checks cover corrected-image links, 100%
  zoom, pan, pixelated display and flicker.

Resource wrapper measurements on i265:

```text
full quality replay: rc=0 17s | peak-RSS 0.22GiB | min-avail 23021MiB | peak-load 1.36
native API replay:   rc=0  1s | peak-RSS 0.01GiB | min-avail 23130MiB | peak-load 0.91
GPU witness rerun:   rc=0  5s | peak-RSS 1.29GiB | min-avail 22774MiB | peak-load 0.58
strict WASM checks:  rc=0  7s | peak-RSS 0.09GiB | min-avail 23496MiB | peak-load 0.54
browser repair:     rc=0  4s | peak-RSS 0.18GiB | min-avail 22587MiB | peak-load 0.49
```

No larger-size, deployment-latency or fresh held-out claim is made. Published
routing numbers predate this correction and have not been rescored in full.

Artifacts on i265, `/mnt/v/output/zensr/sr-bank-20261110/`:
`projection-fixed-bank-v1/`, `projection-fixed-report-v1/`, `fixed-witness-v2/`,
`fixed-gallery-v1/`, all three `fixed-wasm-*-v1/` packages, native/browser
receipts and frozen binaries. `paired-cells.parquet` and
`category-quality.parquet` retain scalar results without maps. Witness v1 is
also preserved; v2 records the explicit orchestration revision separately from
the runtime revision. `FIXED_RUNTIME_RECEIPT.json` links the completed stages.

Verified retained archive: `output/zensr/sr-bank-20261110-projection-fix-v1.tar`,
188,467,200 bytes; SHA-256
`11e7b9fa581e47419ec22aded9eb66d1ac3b68fc2b951c2e9b6a9ddc41095c98`.
Sender/receiver hashes match; file and directory fsync completed, and three
seeded random corrected-image members passed independent readback.
The archive includes candidate images, scalar tables, browser packages, frozen
binaries, prior browser assets, live gallery and full validation logs.
`FIXED_BACKUP_RECEIPT.json` records transfer and readback identities.
