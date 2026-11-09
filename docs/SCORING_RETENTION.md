# Scoring retention

Routine scoring retains scalar measurements, encoded JPEG candidates, source
references, routing features and reproducible source/model/codec/metric identities.
Retain generated upscaled images with lossless compression, together with their
pixel and encoded-file hashes. The RGB8 image retained must be exactly the image
passed to IQA; compression must not change its pixels. Floating-point working
buffers can be released after that artifact is safely written. Diagnostic metric
maps are off by default. A separate map diagnostic requires explicit opt-in,
a bounded witness set and a byte budget.

The November 9 clarification distinguishes upscaled image retention from IQA
map retention. The first external SR comparisons incorrectly applied a
scores-only policy and never persisted their upscaled outputs. Their scores and
hashes remain intact, but the images require regenerated inference. Historical
run identities keep their actual retention labels. New SR comparisons use
lossless PNG output retention; this does not authorize any further deletion.

The routing-study and QAT evaluators record `retention=scores-only-v1` in their
run identities. Their historical output-pixel/map path columns are empty; this
is an explicit contract, not a missing-file fallback. References and JPEGs remain
retained. Input verification uses that declared artifact set; legacy runs still
require their original artifact fields. The frozen old evaluators are not rebuilt
or silently resumed under the new policy. A diagnostic opt-in mode is not supplied
by these routine evaluators.

## Existing completed routing studies

The earlier implementation interpreted the November 8 cleanup request as covering
scored decoded/repaired pixels as well as diagnostic maps. The owner clarified
on November 9 that the requested drop concerned IQA maps, not upscaled images.
The existing tool and receipts describe historical behavior; they are not
standing authorization to remove generated images. The tool plans from completed cell roles and
excludes every reference/JPEG input, including reference/output deduplication.
It never deletes unreferenced files, source references, encoded inputs, models,
scores, coverage or tool identities. A completion marker is required; modern
runs must match their full coverage ledger. Early completed pilots without a
ledger need the explicit `--legacy-scored-cells` option, and their receipts
do not certify grid coverage. Only their recorded scored outputs are removed.

```sh
just publication-routing-retention plan RUN PLAN --tables TABLE_SNAPSHOT
just publication-routing-retention apply RUN PLAN
```

The immutable plan records candidate paths, inode/size/mtime signatures and
allocated bytes. Apply validates the entire plan before deleting, records intent
durably, and supports resuming that same plan. It refuses live routing writers,
escaping paths, output symlinks, changed inputs and altered plans. It does not
perform another pixel-artifact audit or change score values.

`ARTIFACT_RETENTION.json` records progress and completion. An existing
`AUDITED.json` moves to `audit-history` because it describes the pre-removal
artifact set. Immutable score-table snapshots retain their historical audit
identity; they no longer imply that all original pixel artifacts remain present.
Metrics and maps can be regenerated from retained inputs using the frozen
implementation/environment. Direct inspection of removed outputs requires that
recomputation.
