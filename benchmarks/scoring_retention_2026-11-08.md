# Scored-output retention — 2026-11-08

The owner authorized discarding decoded/repaired RGB buffers and diagnostic
metric maps after scoring. All ten completed runs in the original November 4
routing-study directory were pruned. Unfinished runs and unreferenced files
were preserved.

| Scope | Removed files | Removed allocated bytes | Retained JPEG/reference inputs |
|---|---:|---:|---:|
| Primary development | 932,129 | 378,724,683,776 | 235,638 |
| Nine completed pilots | 2,174 | 3,799,064,576 | 673 |
| Total | 934,303 | 382,523,748,352 | 236,311 |

The directory measured **390,309,404,672 allocated bytes before removal** and
**7,985,520,640 afterward**: **390.31 GB → 7.99 GB**, using decimal GB. The net
directory reduction differs from removed-file bytes because the deletion plans
and receipts are retained. These are filesystem allocated-byte observations,
not a claim about physical free space attributable to this task.

All retained artifact counts match the plans' protected-input counts. The
primary run retains 235,154 unique JPEG files and 484 reference RGB buffers;
JPEG deduplication means file count differs from cell count. No Butteraugli
maps remain in the ten pruned artifact sets. Current metadata hashes match
the pre-removal identities. The primary compressed score-table archive still
has SHA-256 `0ff5a241849430ecad22b9bc911c94d15ed8d10d7844482b80d194d1b4fd4bfb`.
[Machine-readable results](scoring_retention_2026-11-08.json).

This follow-up read directory entries and scalar/identity metadata, not pixel
contents. It did not restart the cancelled pixel-artifact audit. Historical
artifact-audit markers moved to `audit-history`; `ARTIFACT_RETENTION.json`
records the removed set and completion. The earliest four-cell pilot predates
coverage ledgers and used explicit legacy-scored-cell admission; its receipt
does not certify full grid coverage. The other nine runs bind full coverage.

## Future scoring

The routing-study and QAT evaluators now request score-only Butteraugli results
and do not serialize decoded/repaired pixels. Their identities declare
`scores-only-v1`; diagnostic path columns are explicitly empty. Reference inputs,
encoded JPEGs, scalar measurements, routing features and tool/model identities
are retained. Diagnostic maps require a separate explicitly budgeted witness run.
[Retention contract](../docs/SCORING_RETENTION.md).

Routing and QAT evaluator tests and clippy pass. The 40 routing integrity
regressions pass in the pinned Python environment; ten retention regressions
cover protected references, changed-plan refusal, resume, legacy admission and
batch boundaries. Existing quality thresholds and restoration code are unchanged.
The first local routing build encountered incomplete cached `cc` sources; an
isolated Cargo source cache built and tested successfully. The system Python
had incompatible scientific-package versions; the pinned environment passed.

## Reproduction and retained records

Implementation commits: `80748c5a` (role-based pruning), `930476d7` (score-only
evaluators), `155ce5e1` (explicit legacy-pilot admission).

```sh
just publication-routing-retention plan RUN PLAN --tables TABLE_SNAPSHOT
just publication-routing-retention apply RUN PLAN
just publication-routing-retention-inventory STUDY_ROOT
```

The [original study pointer](publication_routing_2026-11-04.pointer.md) locates
the retained inputs and tables. Its root contains `retention-plan-primary-v1`,
`retention-plans-pilots-v1`, per-run receipts and the exact pruning-tool snapshots.
The primary output manifest has SHA-256
`26a19e630bbb691984aa624f8ccfefda2a5be97876171bc2bf56abd968e1d41c`.
Operational logs and the independently replayed inventory report are retained
on i265 at `/mnt/v/output/zensr/routing-retention-20261108/`.

Those operational records were mirrored to
`tower:/mnt/disk1/coefficient/output/zensr/routing-retention-20261108/operational-record-v1.tar`:
81,920 bytes, SHA-256
`d1acb9782a99261c4d25d4b72f815c25d04914533d7e0acdde7061f3bc56887e`.
Sender/server lengths and digests matched; file and directory synchronization
completed. The complete per-file removal manifests remain at the study root.
