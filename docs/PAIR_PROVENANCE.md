# Recovering identities for existing training pairs

The old v4 dataset ledger records encoder settings but omits source IDs and
crop coordinates. `tools/recover_pair_sources.py` reconstructs those fields
from the historical crop-selection recipe, then requires exact agreement with
the existing arrays. It does not overwrite arrays or regenerate JPEG inputs.

## Output

- `pairs.tsv`: one row per saved array index, identifying the imazen-26 ID,
  canonical source path, crop rectangle, canonical split, effective split,
  optimizer-training/internal-validation partition, encoder settings and
  SHA-256 hashes of source bytes and both saved RGB crops.
- `sources.tsv`: one row per contributing source, with its ID, dimensions,
  actual and canonical-manifest byte hashes, split labels and pair counts.
- `summary.json`: coverage, canonical exposure counts, input/output hashes,
  corpus revision, tool identity, environment and limitations.
- `progress.json`: stage, completed/required counts, PID, last update and any
  failure. Failed attempts retain partial files and never acquire final manifest
  names merely because reconstruction started.

The separate `tools/verify_pair_sources.py` checks the final manifest by
loading each source once and taking its recorded crop rectangles. It uses no
replay seed. It compares every saved target pixel, checks every LR/HR crop hash,
and joins canonical IDs and split labels independently of the derived split.
Its output includes `verification.json` and `sources-with-links.tsv`, retaining
all canonical source/render URLs and the corpus licence labels.

Crop coordinates are zero-based in the historical decoder's returned raster;
source dimensions are decoded dimensions. They do not necessarily describe the
stored raster before orientation handling. Use the recorded decoder path when
recreating crops rather than applying an additional orientation transform.

## Reproduce

Use an environment with the historical decoder dependencies (NumPy, OpenCV,
PyTorch, Pillow and pillow-heif). Run through the host resource wrapper with
explicit thread and memory limits. Give each attempt a new output directory;
existing results are never overwritten.

```sh
ZENSR_PYTHON=/path/to/python just test-pair-recovery

ZENSR_PYTHON=/path/to/python just recover-pair-sources \
  --dataset /path/to/existing-v4-pairs \
  --corpus /path/to/imazen-26 \
  --model-meta models/adopted/dejpeg_rt24i_canon4/meta.json \
  --output /path/to/new-reconstruction

ZENSR_PYTHON=/path/to/python just verify-pair-sources \
  --manifest /path/to/new-reconstruction \
  --dataset /path/to/existing-v4-pairs \
  --corpus /path/to/imazen-26 \
  --output /path/to/new-coordinate-check
```

## What a match establishes

A complete run requires all saved HR targets to equal their proposed source
crops byte for byte, followed by agreement with every recorded encoder,
quality, subsampling, generation-count and clean-pair draw. Clean LR pairs
must equal their HR targets. All dataset inputs and contributing source bytes
are rehashed before completion. The coordinate checker additionally consumes
the emitted manifest without recreating the RNG sequence.

This is **reconstructed provenance**, not a contemporaneous generation log.
It identifies a complete replay consistent with the saved pixels and recipe;
it does not prove a unique historical path when different source files have
identical pixels. Non-clean LR images are bound by hashes, not recreated with
historical encoder binaries. Severity and uncertainty channels remain recorded
inputs, not newly verified encoder outputs.

Source IDs and crop rectangles allow actual canonical-bucket exposure to be
counted or canonical-only subsets to be selected. They do not remove exposure
from existing weights or establish a compliant retrain. See the
[model selection-policy record](MODELS.md#training-and-held-out-membership).
