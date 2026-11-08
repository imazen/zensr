# Scoring retention

Routine scoring retains scalar measurements, encoded JPEG candidates, source
references, routing features and reproducible source/model/codec/metric identities.
Decoded and repaired RGB buffers are temporary scoring inputs; discard them after
their scores are recorded. Diagnostic metric maps are off by default. A separate
diagnostic run requires explicit opt-in, a bounded witness set and a byte budget.

The routing-study and QAT evaluators record `retention=scores-only-v1` in their
run identities. Their historical output-pixel/map path columns are empty; this
is an explicit contract, not a missing-file fallback. References and JPEGs remain
retained. Input verification uses that declared artifact set; legacy runs still
require their original artifact fields. The frozen old evaluators are not rebuilt
or silently resumed under the new policy. A diagnostic opt-in mode is not supplied
by these routine evaluators.

## Existing completed routing studies

The owner authorized dropping their scored decoded/repaired pixels and diagnostic
maps on November 8, 2026. The retention tool plans from completed cell roles and
excludes every reference/JPEG input, including reference/output deduplication.
It never deletes unreferenced files, source references, encoded inputs, models,
scores, coverage or tool identities. Incomplete grids cannot be pruned.

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
