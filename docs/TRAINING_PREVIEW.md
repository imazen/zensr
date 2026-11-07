# Train from the pair preview

The draft data release preserves all 48,000 existing pairs. Two archives contain
39,895 rows whose **source IDs are canonical train**. They retain 39,503
optimizer-training rows and a separate 392-row internal validation tail.
A third archive contains 8,105 canonical validation/test rows and is excluded
by the default assembler. This subset supports a compliant source-ID boundary;
it does not retroactively change the original canon4 weights or establish
content-family independence across canonical IDs.

## Clone, download and assemble

Use Python 3.12 and a current PyTorch build for your CPU or CUDA system. The
preview records the [tested environment](../benchmarks/preview_training_environment_2026-11-07.json);
installing dependencies does not promise
bit-identical retraining across hardware or package versions.

```sh
git clone https://github.com/imazen/zensr.git
cd zensr
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r tools/training-requirements.txt

# Draft release assets require repository write access while unpublished.
gh release download data-preview-2026-11-07 --dir ../zensr-preview \
  --pattern bundle-index.json --pattern 'canon4-canonical-train-*.tar.zst'
python tools/pair_release.py assemble \
  --index ../zensr-preview/bundle-index.json \
  --output ../zensr-canonical-pairs
```

Install the `zstd` and `tar` command-line tools before assembly. Archives and
their members are SHA-256 checked before use. The assembler preserves original
pair indices, enforces canonical `train` source labels, and checks that internal
training/validation source IDs do not overlap. It never rebuilds JPEG encodes.

## Train

```sh
ZENSR_DATA=../zensr-canonical-pairs \
ZENSR_SCALE=1 ZENSR_NF=24 ZENSR_NC=8 ZENSR_QBOOST=3 \
ZENSR_CKPT_EVERY=5000 ZENSR_OUT=dejpeg_rt24_canonical_preview \
python tools/train_people.py 200000 48 3e-4
```

The trainer reads `val_tail` from the assembled dataset metadata (392 here,
512 for the original dataset). Training a new subset produces a new model;
it does not reproduce the historical canon4 artifact. Follow the host's resource
budgeting rules. For a CPU smoke check, add `ZENSR_DEV=cpu ZENSR_COMPILE=0` and
use one step with batch size one instead of the full run.

All four pair arrays, original-index maps and codec settings are included.
Recovering source pixels does not require downloading imazen-26. Source IDs
can be joined to the pinned corpus metadata indexes linked in `bundle-index.json`.
The more detailed local reconstruction retains source filenames and URLs.

## Scope of the preview

The ID-based asset metadata contains no machine logs or descriptive source
filenames. RGB arrays contain decoded crops and no EXIF. Their visible content
has not been reviewed for personal information. Corpus licence labels, including
`screenshot-unverified`, remain in the source catalog. Keep the release as a
draft until content and redistribution review are complete.

The original training log spans **35 minutes 34.6 seconds**, from its filesystem
creation at 2026-11-04 02:14:26.060803393 UTC to its last write at
02:50:00.650332986 UTC. It records CUDA/bfloat16 and 200,000 steps. This is a
log-timestamp span, not a separately recorded total-process stopwatch.
