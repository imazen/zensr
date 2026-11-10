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
