# zensr — JPEG repair and CPU super-resolution

## JPEG repair status — 2026-11-07

The ×1 repair implementation and owner-approved default are committed. The
implementation passed CI on Linux x86/ARM, Windows x86/ARM, macOS ARM and i686,
including formatting and Clippy. Independent model replays, selected-path pixel
parity and metadata-invariance checks are retained.

This is a locked-source integration, not a crates.io release. Quality numbers
below are limited to 512 pixels and reuse an inspected cohort. Arbitrary-size
execution is implemented; quality across sizes and deployment latency remain
unestablished. See [readiness and remaining work](docs/REPAIR_READINESS.md).

Default `api::Restorer` Fidelity requests use the owner-selected decoded-feature
router for RGB 420/444 JPEGs at arbitrary resolutions (minimum dimension eight
pixels). It analyzes native windows without resizing and restores the original
dimensions. On 140,940 reserved **512-pixel** cells it gives **+2.417 mean SSIM2
and 0.566% harm**, both origin-balanced, restoring 79.75% of cells. This
secondary check reuses the inspected cohort. Harm uses the quality-dependent
loss curve; the worst loss is 18.534 SSIM2. Pixel-artifact auditing was cancelled by the owner.
[Metadata-independent default, comparison and limitations](benchmarks/routing_feature_ablation_2026-11-06.md).

Explicit `Routing` overrides, Conservative intent, unsupported JPEG layouts and
`--no-default-features` builds use the calibrated coefficient/quality routing.
The heads exclude file bpp and image dimensions, so opaque metadata padding
cannot affect them through file size. The `chooser` analysis dependency is
enabled by default for decoded routing;
the older content-class classifier is not an input to the new linear router.
Quality at other resolutions and runtime cost have not been established.
Low-level `restore_jpeg` remains the configurable
restoration pipeline. [Default routing implementation](docs/DEFAULT_ROUTING.md).

Ship format is f16. Goldens are generated through the f16 roundtrip and tested
in Rust on whole and tiled paths. The library accepts caller-loaded models;
its current routing estimates are calibrated for canon4, not arbitrary weights.

## Workspace

- `zensr-zenjpeg`: JPEG probing, routing, decoded-pixel repair, guards and
  coefficient projection. The public repair call is ×1 only.
- `zensr-micro`: CPU model inference, SIMD kernels and tiled processing;
  `zensr-micro-abi` provides a separate C entry point for SPANF ×4.
- `zensr-bench`: quality evaluation, model comparison and profiling tools.
- `zensr-gpu-spike`: research CUDA/wgpu kernels; no production GPU backend.

The longer-term upscaling and training plan is [PLAN.md](PLAN.md).

## Historical model comparisons

**Low-level ×1 repair pipeline** (`zensr-zenjpeg::restore_jpeg`): probe → decode policy
(Knusperli only Annex-K q≤9.5) → guarded ×1 model → S10 quantization-consistency
projection (family-calibrated relative and absolute slack; not a bit-exact
re-encoding guarantee) → output pixels. Repair → ×2 chaining exists in the
evaluation harness as a separate operation.

| tier | model | params / f16 size | ssim2 gain q15/35/55/75/90 | s/MP @12T |
|---|---|---|---|---|
| historical quality tier | dejpeg7_graphics | 595k / 1.16 MB | **+7.78/+6.11/+4.60/+3.19/+1.06** | 5.3 |
| realtime | **dejpeg_rt24i_canon4** (four-encoder retrain, 200k steps) | 43,203 / **86,406 bytes** | **+6.542/+4.246/+2.473/+1.308/+0.593** | not measured |
| low-q graphics route | dejpeg9_gfxycc | 595k / 1.16 MB | **+2.72/+1.45/+1.09/+0.78/+0.21 OVER dejpeg7 on graphics** (paired, 81/81/89/81/67% of files); **negative on photographs from q55 up** — route it, do not default to it | 5.3 |

The realtime row is the 2026-11-04 canon4 measurement: 63 held-out PNG
references, turbo 4:2:0, per-file median of projected output minus identity,
with the identity gate disabled. Auto routing skips some of those restores.
The quality-tier row retains its older measurement; it is not a contemporaneous
head-to-head comparison. [Calibration and skip tradeoffs](benchmarks/ship_rt24i_canon4_2026-11-04.md).
[Runtime dependency chain](docs/PRODUCTION_DEPENDENCIES.md).

> **Re-measured 2026-09-08 on the canonical corpus.** The previous figures
> (+10.65/+5.64/+3.53/+2.03/+0.95 and +6.88/+3.36/+2.02/+1.15/+0.32) were produced
> against the wrong imazen-26 and a split that was never a split; both are gone.
> The models held up — they are *lower at q15* and *higher at q35–q90* than
> before — but no old number was reproducible, so all of them are replaced rather
> than defended. Method and the full comparison:
> `benchmarks/rescore_canonical_2026-09-08.md`. Note also that
> `dejpeg7_graphics` inherits from a fine-tune chain that trained on 20 corrupted
> ground-truth pairs (`docs/MODEL-PROVENANCE-AUDIT.md`); the committed realtime
> weights are outside that lineage, and a retrain is queued for the corpus swap
> regardless.

Historical quality-tier/rt24g comparison: **canonical imazen-26** (`github.com/imazen/imazen-26`),
held-out split (validate ∪ test of `eval_split/imazen26_effective_split.tsv`,
**0 training files scored**), PNG references, libjpeg-turbo 4:2:0, **per-file
median** gain over the plain decode, n=36/cell across 21 content classes, shipped
pipeline output (post-projection). Win fractions 97/100/100/94/84%
(dejpeg7) and 98/100/94/84/67% (rt24g) on the full 63-file sample. Rows:
`benchmarks/clean_ladder_2026-08-03.pointer.md`.

> Corrected 2026-08-03, in **two opposite directions**. Absolute gains were
> *understated* — the previous +7.1/+5.0 at q15 were measured against a partly
> JPEG-sourced eval split, which penalised the model for removing artifacts
> that were present in the reference. But realtime-as-a-fraction-of-quality was
> *overstated*: measured **57–65%**, against the published 68–79%. "The
> contamination understated everything" is too simple.


Runtime: safe-Rust explicit SIMD throughout — one `#[magetypes]` f32x8 kernel
family (AVX2/SSE4.2/NEON/WASM128/scalar) + a hand `_v4x` AVX-512 f32x16 variant,
packed weights, tiled multithreading with exact halos, strided rows. Measured
efficiency: realtime tier ~410 GFLOPS aggregate @12T (near practical peak);
quality tier ~225 GFLOPS (cache-bound at nf=64 — open rungs: strip-fused layer
scheduling, Winograd F(2,3)). No production GPU path. The existing CubeCL CUDA/wgpu spike is a research
baseline, described in `SYSTEMS.md`.

Key falsifications (full registry in SYSTEMS.md): input conditioning (scalar &
dmap), YCbCr-native as general pipeline (3 designs incl. de-confounded scratch
pair), lattice-aware chroma arch above q35 (oracle+lattice decomposition),
guided chroma upsampling (GT-guide JBU loses to bilinear), zero-bias weight
compression for conv kernels, Boundary4Tap at any q.

### Distillation approach (verified against literature 2026-07-28)

Historical experiment (not the canon4 training recipe): **output distillation**: the student trains on quality-tier teacher
outputs instead of GT (S9). Measured: +0.41 ssim2 paired over direct GT
training at 113k, and the recipe **saturates at 43k params** (rt24d == rt32d
quality at 3.2× speed). A zenpapers sweep (500-record SR manifest; 14 KD
papers 2019–2026: IMDN/RFDN lineage, FAKD-style feature distillation, DVMSR,
DSCLoRA, edge GAN-KD, INT8 teacher-guided QAT) shows feature/affinity KD
claims of small additional gains on tiny students — but our student already
saturates its capacity under output-KD, so feature-KD is queued as a
falsification rung, not adopted. What makes ours *documented*: paired
same-seed controls, committed TSVs, and per-model repro scripts.

## Measured results (2026-07-22, this box, single-thread)

| Thing | Number |
|---|---|
| SPANF x4 via tract 0.23.4 | 3.27 MP-out/s (128², 80 ms) · 3.33 (256², 315 ms) |
| TSR / EFDN / NanoSR via tract | 2.0 / 1.1 / 1.25 MP-out/s (128²) |
| **micro-v4x vs tract, interleaved paired bench (quiet box)** | **128²: 57.6 vs 71.5 ms (1.24× faster); 256²: 233 vs 292 ms (1.25×); 64²: 18.0 vs 16.2 ms (near-parity)** |
| AVX-512 v4x vs AVX2 v3 (same kernels) | **1.33–1.42× speedup** |
| **Multithreaded tiling (512²→2048², exact 16-px halo)** | **15.5 MP-out/s @8 threads · 17.8 @16 · 18.2 @24 (230 ms for a 16.8 MP ×4 output; tile 112-128)** |
| zensr-micro-abi cdylib, scalar-only | 271,432 B (265 KB) |
| **zensr-micro-abi cdylib, full SIMD dispatch** | **413,936 B (404 KB)** (post-recovery, unfused kernels + gate) (v4x AVX-512 + v3 AVX2 + mandatory scalar fallback; `tier_v4` feature off by default; `incant!` always links scalar) |
| correctness vs PyTorch golden (every tier v3/v4/v4x) | max_abs 8.6e-6 (PASS) |
| **f16 weights (297,696 B)** | **59.2 dB ramp / 75–76 dB photo PSNR vs fp32 — TRANSPARENT, ship it** |
| int8-pc weights (152,400 B) | 16.8 dB ramp / 35–36 dB photo — **NOT viable** (see below) |
| SPANF weights fp32 | 593,152 B |

Full grids: `benchmarks/tract_cpu_2026-07-22.tsv`, `benchmarks/quant_accuracy_2026-07-22.tsv`.

## Weight-compression verdicts (tools/quant_accuracy.py, torch sim + Rust decoders agree exactly)

- **f16: production-safe.** Worst case 0.3% max-error on photos, 2.5% of pixels shift ±1 u8 step
  (75-76 dB). Rust decoder (safe bit-twiddle, subnormal-correct) in `decode_f16_weights`.
- **int8 post-training quantization: broken for SPANF** (35 dB photo, 90%+ u8 mismatch on ramp).
  The gate-exemption probe (quantize all but the σ-gate-feeding convs) changed nothing —
  sensitivity is distributed across the conv chain, so mixed-precision doesn't rescue it.
  int8 for SPAN-class requires QAT (MAI recipe) or an int8-by-design arch (ABPN/ECBSR/ETDS).
- int6/int4/k-means-256 codebook: 5-21 dB, dead on arrival.

## SIMD implementation notes

- One `#[magetypes(v3, neon, wasm128, scalar)]` family (f32x8) + hand `#[arcane]` `_v4x` (and
  feature-gated `_v4`) on native 512-bit f32x16; kernels macro-instantiated per width; whole
  forward inlines into one target_feature region per tier; `incant!` dispatches once per call.
- **magetypes gotcha (root-caused here): `recip()` is a Newton-refined approximate reciprocal —
  `recip(inf) = NaN` (inf·0 in the refinement step). Sigmoid after `exp_midp` saturation MUST use
  exact division.** Real SPANF pre-activations reach ~±100, so exp(-v) overflowing to inf is by
  design, not an anomaly. Symptom was 97% NaN output while `f32::max`-based diff stats looked
  clean — always NaN-check in diff stats.
- Archmage tier map (current, 0.9.28): v3 = AVX2+FMA (native 256), v4 = AVX-512 base,
  v4x = AVX-512 extended; `F32x8Convert` (transcendentals bound) is NOT implemented for v4/v4x
  tokens — use f32x16 there.


## imazen-26 quality eval (2026-08-03, n=8/subcorpus; SPANF vs Lanczos-up)

Paired per file — median of `spanf − lanczos` over images, with the fraction of
files each metric improves. Not the difference of per-method medians, which
compares each image against the *other method's distribution* and reports
losses that no image experienced.

| subcorpus | ΔPSNR | win | ΔSSIM2 | win | Δbutteraugli n3 | win |
|---|---|---|---|---|---|---|
| documents | **+2.44 dB** | 1.00 | **+26.86** | 1.00 | **−3.354** | 1.00 |
| maps | **+2.16 dB** | 1.00 | +11.55 | 1.00 | −1.548 | 1.00 |
| art-scans | +0.54 dB | 0.75 | +12.63 | 0.88 | −0.499 | 0.62 |
| textures | +0.46 dB | 1.00 | +8.13 | 0.62 | −0.690 | 0.75 |
| screen | +0.51 dB | 0.75 | +6.69 | 0.75 | −0.599 | 0.88 |
| people | +0.77 dB | 1.00 | +6.38 | 1.00 | −0.609 | 0.88 |
| photos | +1.16 dB | 1.00 | +5.49 | 0.88 | −0.702 | 1.00 |
| renders | +0.92 dB | 1.00 | **−1.81** | 0.38 | −0.924 | 0.88 |
| **all** | **+0.90 dB** | 0.94 | **+8.45** | 0.81 | **−0.837** | 0.88 |

(Butteraugli: negative is better.)

**SPANF wins 8/8 subcorpora on PSNR and butteraugli, 7/8 on SSIM2**, dominating
text and line content. `renders` is the one disagreement: PSNR improves on all 8
files and butteraugli on 7, while SSIM2 says the median image gets worse — a
metric disagreement on synthetic content, not a demonstrated regression.

References are 100% PNG (`/mnt/v/imazen-26-clean`), files pinned to
`eval_split/imazen26_eval_files.tsv`. (Both are gone: the 2026-09-08 corpus
repoint replaced the pin with the canonical split, and `imazen-26-clean` derived
from the deleted acquisition root and has not been rebuilt. Numbers below predate
the repoint and are provisional. `docs/CORPUS-REPOINT-IMPACT.md`.) Harness: `eval` bin (zenpng/zenjpeg
decode → CatmullRom ×4 down → {spanf, lanczos, catmullrom} up →
psnr/ssimulacra2/butteraugli-n3 vs HR). Caveat: LR degradation is linear-light
CatmullRom, not SPANF's encoded-space-bicubic training distribution — SPANF
scores are conservative. Rows: `benchmarks/sr_pinned_2026-08-03.tsv`; analysis
`benchmarks/sr_pinned_2026-08-03.md`.

> Superseded 2026-08-03: the previous table reported the difference of
> per-method medians, which produced a "textures −1.7 dB / −1.7 SSIM2 (SR loses
> on stochastic detail)" entry. Paired on that same data, textures is **+8.79
> SSIM2 with SR winning 7 of 8 files**; the −1.7 compared two different images.
> Per-file values are unchanged and reproduce exactly — only the statistic did.

## Reproduce

```sh
just export      # NTIRE25 ckpts -> models/*.onnx (needs torch; box GPU used by SPANF init)
just dump        # SPANF weights + golden -> models/*.raw
just bench       # tract grid -> benchmarks/
just verify      # golden-gate zensr-micro + timing
just size        # build + report micro-abi cdylib size
```

## Perf status (2026-07-22, RESOLVED on quiet box)

The decisive fix was the **overlapped final vector tile**: scalar edge columns had covered
25% of width at 64² (each doing the full cin×ky reduction). With edges reduced to 2 columns,
the interleaved paired shootout (benchmarks/shootout_2026-07-22.txt, load ~4.4, min-of-14):

| size | tract 0.23.4 | micro-v4x | ratio |
|---|---|---|---|
| 64²→256² | 16.2 ms (4.04 MP/s) | 18.0 ms (3.64) | 1.11× slower (per-call Packed::build) |
| 128²→512² | 71.5 ms (3.67 MP/s) | **57.6 ms (4.55)** | **1.24× faster** |
| 256²→1024² | 291.9 ms (3.59 MP/s) | **233.4 ms (4.49)** | **1.25× faster** |

AVX-512 (v4x, f32x16) vs AVX2 (v3, f32x8) on identical kernel source: **1.33–1.42×**.

## Multithreaded tiling (landed 2026-07-22)

`SpanfModel` (packed once, `Sync`) + per-thread `Scratch` + `spanf_x4_tiled(model, input,
h, w, threads, tile /*0=auto(128)*/)`. Exact 16-px halo (= receptive field: 16 chained 3x3
convs); tiled output matches whole-image at the identical 6.557e-6 vs golden; seam test
covers non-multiple dims at 1+3 threads. std::thread::scope + atomic tile counter + mutex
paste — zero new deps. Sweep (512² in → 2048² out, quiet box, min):

| tile | grid | t=1 | t=4 | t=8 | t=16 | t=24 |
|---|---|---|---|---|---|---|
| 112 | 5×5 | 2.63 MP/s | 9.44 | 15.50 | 17.75 | **18.21** |
| 128 | 4×4 | **3.01** | **9.87** | 15.05 | 17.37 | 16.56 |
| 224 | 3×3 | 2.98 | 7.86 | 9.10 (granularity-starved) | 9.41 | 8.95 |

Small tiles win on cache (per-thread scratch: 26 MB @160-ext vs 66 MB @256-ext) AND
granularity. **Arbitrary dims verified** (2026-07-22): SIMD-vs-scalar matrix down to 1x1 and
1-px strips, tiled edge tiles with 1-px cores, and torch goldens at 7x5 / 1x9 / 17x18 through
all three paths (<=2e-6) — any h,w >= 1 is supported and tested.

**Fusion + stride-padding post-mortem (2026-07-22):** both were tried and REVERTED. The `Post`
store-fusion caused a 26x regression, mechanism IR-verified: rustc strips `alwaysinline` from
functions containing `#[target_feature]` calls (soundness interlock), so inlining into the
arcane happens only in rustc's MIR inliner — whose size caps `#[inline(always)]` does not fully
override. Fusion's exp polynomials pushed conv3x3's MIR over the cap → rejected → emitted
featureless (`{ nonlazybind uwtable "target-cpu"="x86-64" }`, no alwaysinline) → LLVM legally
cannot inline `+avx512f` intrinsics into it → `call _mm512_fmadd_ps` in the hot loop. Fast
builds have no conv3x3 define in IR at all (fully MIR-inlined). Rule of thumb: keep
arcane-adjacent kernel bodies small; hoist transcendental math into separate per-tier fns. And +16 stride padding at 4-KiB
planes measured **2x slower** at 128²/256² — the L1-set-aliasing hypothesis is falsified for
this access pattern. The cs channel-stride plumbing is kept at identity stride. Row-band
tiling was evaluated and rejected (50–377 MB per-thread scratch on wide images — 2D tiles
scale better). Practical: a 1 MP image → 16 MP ×4 output in ~230-240 ms on this 7950X
(vs ~3.7 s single-thread untiled). Note: tract could be externally tiled the same way —
the engine-level single-thread comparison (micro 1.25× faster) is the apples-to-apples one.

## Remaining headroom (queued)

1. Fuse SiLU/gate into conv stores; pixelshuffle straight from the conv2 tile.
2. Plane-stride padding (+16 floats) to break L1-set aliasing at power-of-2 sizes.
3. cargo-bloat diet toward sub-300 KB full-SIMD; single-tier builds for known fleets.
4. Row-band tiling (halo only left/right) for lower halo tax on wide images.
5. If int8 compute is ever wanted: QAT via MAI recipe, or switch arch to ECBSR/ETDS class.

Status honestly stated: correctness (all tiers), size menu, f16 path, and the tract
comparison are all measured; 64² still trails tract by 11% (per-call weight packing —
fix queued); nothing is product-wired (no zenpipe/imageflow integration, no zencodec
surface). Weight licensing: NTIRE2025_ESR repo is MIT but per-team weight terms are
unaffirmed — fine for experiments, diligence (or retrain on our corpora) before shipping.

## Building

```bash
cargo build --release -p zensr-micro -p zensr-zenjpeg   # the restoration crates
cargo build --release -p zensr-bench                    # the measurement harness
```

Two dependencies are pinned to git revisions rather than crates.io releases,
because the versions we need are unpublished: `zenjpeg` 0.9.0 (0.8.4 keeps
`DeblockMode` crate-private, which the deblock-policy code needs) and
`zenanalyze` 0.2 (the content-class features the router was trained against).
Both repos are public, so a plain `cargo build` resolves them. When those
crates publish, the pins become ordinary version requirements.

If you have the sibling `zenjpeg` checkout, `just cargo-local` writes a
gitignored `.cargo/config.toml` that points the dependency at your working
tree instead of the pinned revision.

`zensr-bench` additionally needs `cjpeg` (libjpeg-turbo) on `PATH` for the
encode arms, and the imazen-26 corpus, which is not distributed here.

## License

AGPL-3.0-only OR LicenseRef-Imazen-Commercial — the same dual license as the
rest of the zen stack (zenpng, zenresize, zensim). See `LICENSE-AGPL3` and
`LICENSE-COMMERCIAL`.

## Status: pre-0.1 research

Nothing here is released. The measured record lives in `SYSTEMS.md`, the
forward plan and the falsification registry in `ROADMAP.md`, and the proposed
public surface in `docs/API_DESIGN.md`.

Numbers predating 2026-07-31 were measured against a partly JPEG-sourced eval
split, and on 2026-08-02 the file selection was found to admit training images
as well. The tier table and the SR table above have both been re-measured on
clean references with pinned selection (2026-08-03) and are current. Anything
else in this file dated earlier has **not** been re-measured — treat those
figures as provisional and check `ROADMAP.md` §0 before quoting them.

Two eval harness defects caused this and are now fixed at the source: file
selection is pinned to the held-out split (now
`eval_split/imazen26_effective_split.tsv`) by both eval
binaries, and every row records `gt_src` so reference provenance can be audited
after the fact. The shared helpers live in `zensr_bench`; any new eval binary
must use them.

See the [quality-dependent harm comparison](benchmarks/routing_harm_curve_2026-11-04.md)
for routing tradeoffs under a 0.5-loss threshold below SSIM2 90 and a taper above it.
