# Train from the pair preview

The draft data release preserves all 48,000 existing pairs. Two archives contain
39,895 rows whose **source IDs belong to imazen-26's published `train` bucket**.
The authority is `imazen/imazen-26`, revision
`187fbf338ce08e8e6654db7f04ddae58d5263da2`, `manifests/split_map.tsv`.
They retain 39,503
optimizer-training rows and a separate 392-row internal validation tail.
A third archive contains 8,105 rows from imazen-26's published validation/test
source buckets and is excluded
by the default assembler. This subset supports a compliant source-ID boundary;
it does not retroactively change the original canon4 weights or establish
content-family independence across imazen-26 IDs.

[Split authorities and field definitions](SPLIT_DEFINITIONS.md) distinguish this
imazen-26 membership filter from zensr's historical reassignment and the pair
dataset's internal validation partition. Existing archive and CLI names keep
`canonical` for compatibility; here it means the published imazen-26 ID split.

## Setup

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
```

Install the `zstd` and `tar` command-line tools before assembly. Archive/member
SHA-256 hashes are checked. The choice below determines which experiment you run.

## Reconstruct the original canon4 training inputs

**Download all three pair archives, including imazen-26 validation/test sources.**
Historical reproduction requires the actual data canon4 used, even though its
source selection violated imazen-26's published split. This path reconstructs
all **48,000 rows** in original order: **47,488 optimizer-training rows plus
the original 512-row internal-validation tail**. It includes every original
LR/HR, severity and uncertainty array; no JPEGs need regeneration.

```sh
# Draft release assets require repository write access while unpublished.
gh release download data-preview-2026-11-07 --dir ../zensr-preview \
  --pattern bundle-index.json --pattern 'canon4-canonical-*.tar.zst'
python tools/pair_release.py assemble \
  --index ../zensr-preview/bundle-index.json --policy historical \
  --output ../canon4-historical-pairs
```

Use the recorded original trainer revision and recipe for retraining:

```sh
mkdir -p ../canon4-original-source/tools
git show 4a33019:tools/train_people.py > ../canon4-original-source/tools/train_people.py
ZENSR_DATA="$(realpath ../canon4-historical-pairs)" \
ZENSR_SCALE=1 ZENSR_NF=24 ZENSR_NC=8 ZENSR_QBOOST=3 \
ZENSR_CKPT_EVERY=5000 ZENSR_OUT=dejpeg_rt24i_canon4_reproduction \
ZENSR_COMMIT=4a33019 \
python ../canon4-original-source/tools/train_people.py 200000 48 3e-4
```

The original run used torch 2.15.0+cu132, CUDA/bfloat16, torch seed 7 and NumPy
seed 11. Matching data and recipe does not guarantee identical learned weights
across hardware, library versions or numerical settings. The original weights
are separately included in the model draft. This recipe reproduces the
historical experiment; it is not the source-selection policy for a new
imazen-26-compliant model.

## Train a new model using only published imazen-26 training IDs

This is a separate experiment, not reproduction of the existing canon4 model.
Only the first two archives are required. The assembler's default policy selects
them and excludes the imazen-26-validation/test-source archive.

```sh
gh release download data-preview-2026-11-07 --dir ../zensr-preview \
  --pattern bundle-index.json --pattern 'canon4-canonical-train-*.tar.zst'
python tools/pair_release.py assemble \
  --index ../zensr-preview/bundle-index.json \
  --output ../zensr-canonical-pairs
```

The assembler preserves original
pair indices, enforces the published imazen-26 `train` source labels, and checks that internal
training/validation source IDs do not overlap. It never rebuilds JPEG encodes.

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
