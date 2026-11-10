# Small-image repair and SR candidate bank

This offline tool expands the earlier one-image-per-category comparison. It
reuses imazen-26's existing lossless rendered sRGB IMIJ archives, zenresize,
zenjpeg, native guard/projection kernels, and the previously compared CUDA
canon4/SPANF/DSCF implementations. It does not change runtime defaults.

`bank.py fetch` reads the archive inventory and downloads only train/validate
archives over HTTP, verifying their recorded sizes and SHA-256. No NFS path is
used. `freeze` joins every entry to pinned corpus manifests before selecting
images without inspecting quality scores. Render-v4's five-digit export IDs
are mapped to four-digit corpus IDs using its documented leading-`1` mapping.
Both identities remain in the plan.

Selection requires agreement between the published ID split and the successor
family-v2 split, with one representative per declared family. Disagreements,
missing members and unsupported source formats are recorded as exclusions.
Test images are not read. The first bank accepts rendered sRGB RGB8 and Gray8;
grayscale is replicated exactly into RGB. It does not cover HDR or alpha.

The frozen axes are LR long edges 32/64/128, clean input and JPEG qualities
5/10/20/30/40/50/60/70/80/90/100, with 420 and 444 sampling. Integer aspect-fit
preserves the complete frame; HR dimensions are exactly four times LR. The
short side is rounded to the nearest integer, with dimensions recorded.
References and LR inputs use zenresize's linear-light CatmullRom path.

For JPEG inputs, CUDA generates raw canon4 output once. Native `guarded_merge`
and coefficient projection then generate guard-only and half/default/double
slack variants. This is a **forced offline CNN comparison**, not a replay of
the runtime router/deblock policy. SPANF and DSCF each upscale direct, half
repaired and fully repaired RGB8 input. The half repair rounds an equal mixture
of decoded and default-projected RGB8. Original JPEG bytes, references,
decoded input, raw CNN F32, repair variants and final SR RGB8 are retained.

Images are lossless IMIJ entries in one `imij-set` archive per source. Every
image is decoded and compared byte-for-byte before insertion; source archive
reads verify entry checksums. Writers finish, flush and fsync before a completed
origin record is published. Partial files remain visible after a failure.
The bank stops on a 40 GiB archive budget or 80 GiB local free-space reserve.

Quality records use zenmetrics' unchanged PSNR helper from `../sr-blend`, pinned
adopted zensim-A, and SSIMULACRA2 for artifact removal. No metric maps are
requested. Keep scalar scores and images together: later filter/blend studies
can reuse them without inference. Existing canon4 training exposure means
this bank does not establish fresh held-out quality for the base CNN.

The extractor uses the existing zenanalyze-api `Request`, qualified feature
names, `satisfies`, `reuse_for` and `schema_hash`. Feature schema/provenance is
stored once; each vector binds its pixel hash, geometry and processing stage.
Flat inputs can omit the undefined `spectral_slope_y` feature. Each observed
offer schema is retained separately; missing features are not filled with zero.
A model must request a compatible qualified subset. Completed shards can be
reused after an orchestration correction only when the frozen executable,
plan, model and generation dependencies agree; the original generation
identity and source-record hash remain in every reused record.
Matching feature definitions alone does not establish that two offers describe
the same pixels. No compatibility API is added.

Build/test with `just sr-bank-build`, `just sr-bank-test`, and
`just sr-bank-check`. The standalone workspace requires access to the pinned
private `imazen/zqi` repository; ordinary public builds do not include it.
Use `bank.py --help` and each subcommand's help for input paths. GPU generation
requires `CUBLAS_WORKSPACE_CONFIG=:4096:8`, the retained model checkpoints and
the existing CUDA Python environment. Run all generation through the resource
wrapper with explicit CPU/RAM limits. The run identity binds the plan, binary,
model/source bytes, Python packages, PyTorch binary, CUDA/cuDNN versions and precision policy.

The lockfile intentionally binds IMI's zencodec 0.1.27 to zenpixels 0.2.16,
while JPEG uses zenpixels 0.3.1. Their broad dependency ranges permit an
incompatible combination if resolved afresh. Use `--locked`; do not replace
the committed dependency graph without rebuilding both paths.

## Reusing the bank for filter/intensity sweeps

`blend_bank.py` opens the completed bank and runs the same strided pixel
operations as `tools/sr-blend`. Each case retains its references, LR variants
and SR outputs in memory while it tries Triangle, Mitchell, CatmullRom and
Lanczos2 at 0/25/50/75/100% SR, at ×2 and ×4. Both models share the filtered
baseline for a given repair stage. The baseline uses the corresponding
post-repair LR, matching a repair-then-upscale pipeline.

Scalar results stream to Parquet+zstd. Pure-SR endpoints must match the
retained generation pixel hashes and PSNR/zensim-A scores exactly; redundant
filter/model endpoints must also agree. A source finishes only after the
entire expected grid and Parquet row count agree. No blend-image variants or
metric maps are written. The retained lossless source/SR entries reconstruct
any mixture. This command uses a Python environment with PyArrow and Polars; it does not
modify the frozen CUDA environment. Install the pinned scalar-report packages
from `report-requirements.txt`; public CI runs the split/frontier Python
regressions without needing the private image codec or GPU checkpoints.

The zenpredict bake/replay adapter is available for subsequent experiments.
No learned selector has been fitted or adopted from this three-size bank.
Production selection needs denser size/mode evidence than this exploratory
small-image set supplies.

`blend_report.py` retains all settings by split/category/q/sampling/size/model/
scale and reports the train-side Pareto frontier on validation. It also reports
train-best zensim-A and train-best mean-MSE anchors. PSNR aggregation uses mean
MSE so exact matches remain exact. Ground-truth categories/q buckets are offline
strata, not a deployed classifier; no validation scores select settings.

The predictor replay input's `expected_model_schema_hash` identifies the loaded
bake. It is not a gate on the caller's offer: extra/reordered qualified features
are allowed, and `reuse_for` gathers the model's requested subset. Missing or
drifted requested definitions fail. Pixel framing remains the caller's external
responsibility, bound by the bank records here.
