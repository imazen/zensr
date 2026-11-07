# Draft data and model assets — 2026-11-07

Two unpublished prerelease drafts are uploaded:

- [Paired data](https://github.com/imazen/zensr/releases/tag/untagged-6d945203269554d9112c),
  tag intent `data-preview-2026-11-07`.
- [Models](https://github.com/imazen/zensr/releases/tag/untagged-71db29b6982428753dd8),
  tag intent `models-preview-2026-11-07`.

Draft access requires appropriate repository permissions. Neither draft is a
crate publication or approval to publish the underlying content.

## Data layout and training check

| Archive | Pairs | Compressed bytes |
|---|---:|---:|
| Canonical training sources, part 1 | 19,966 | 802,684,752 |
| Canonical training sources, part 2 | 19,929 | 806,185,904 |
| Canonical held-out sources | 8,105 | 241,942,346 |

The first two archives contain only source IDs assigned canonical `train`.
The assembler preserves 39,503 optimizer-training rows and the 392-row internal
validation tail, with no shared source IDs between those partitions. The
third archive includes the 7,985 historical training rows and 120 internal
validation rows whose sources belong to canonical validate/test.

All original pair indices, LR/HR arrays, severity and uncertainty arrays are
preserved. The default assembler excludes the held-out-source archive, verifies
archive/member hashes and enforces source labels. The trainer reads the tail
size from metadata; its original 512-row default is preserved.

A clean source snapshot at `04617a3c990c` assembled the two training-source
archives and completed one CPU optimization step, internal validation and export
of a 43,203-parameter model. The smoke artifact is a test fixture, not a released
trained model. [Clone-and-train instructions](../docs/TRAINING_PREVIEW.md) and
[tested environment](preview_training_environment_2026-11-07.json).

The smoke command used one configured job and an 8 GiB cap. Its resource wrapper
reported `rc=0 33s | peak-RSS 7.64GiB | min-avail 19205MiB | peak-load 16.97`.
The full historical canon4 run was not repeated. Its log timestamps span
**35 minutes 34.6 seconds** for 200,000 CUDA/bfloat16 steps; this is not a
separately measured total-process stopwatch.

## Model scope

The model draft includes **50 adopted model names**, **120 training checkpoints**,
**16 ONNX baselines**, additional raw runtime baselines and **15 fitted routing
estimators**. The main archive captures 452 binary files as 349 unique objects;
the routing archive captures 15 estimators as 14 objects. Catalogs map names,
architecture, scale, hashes and recoverable provenance to byte-preserved objects.

The two model archives have 138,820,393 and 442,995 compressed bytes. This is
archival coverage of the inspected locations, not a new ranking or model-quality
validation. Older models retain their historical data-policy and corrupted-target
lineage limitations. Third-party weights retain their upstream terms.

## Fields, content and material not uploaded

Asset tables use imazen-26 IDs rather than descriptive source paths or machine
logs. NumPy RGB arrays carry no EXIF. Visible pixel content has not been reviewed
for personal information, and `screenshot-unverified` remains in the source
licence catalog. The drafts remain unpublished for content/redistribution review.

`ARTIFACTS-NOT-UPLOADED.json` is attached to the model draft. It identifies:

- Eight arrays from the older two-encoder and zenjpeg-specialist pair datasets:
  **9,486,721,024 bytes**, retained locally.
- **71** original model metadata/command files; portable metadata is supplied.
- **80** synthetic test estimators, not scientific fitted models.
- Other experiment records, predictions, logs and tooling: **147,881 files,
  1,169,570,734 bytes** within the bounded scratch inventory.
- The full publication image/cell corpus, dense-size artifacts and other source
  corpus originals, which are outside that byte count. Source originals remain
  distributed by imazen-26.
- The new one-step smoke artifact and package wheels/driver environment.

Scope: model roots on dev/i265, zensr-prefixed scratch directories through depth
two, and the recorded primary/feature routing fits. **5,619 deeper scratch
directories were not recursed**; other hosts, object stores and unrelated paths
were not inventoried. Those limits are explicit, not a claim that no other
artifacts exist. Detailed path inventories remain outside Git at
`/mnt/v/output/zensr/preview-20261107/TRIAGE-LOCAL.json` on i265. Nothing was deleted.

## Uploaded bytes and verification

All **12 assets** are in `uploaded` state; GitHub's server-side SHA-256 and size
match each local file. Total uploaded bytes: **1,994,914,536**. All members of the
three pair archives and both model archives passed decompression/hash checks.
Both releases were verified as drafts with no publication timestamp.

Small evidence records: [GitHub asset verification](preview_assets_2026-11-07.github.json)
and [archive verification](preview_assets_2026-11-07.archives.json). Large
archives, catalogs and detailed inventories are release assets or retained data,
not Git objects.
