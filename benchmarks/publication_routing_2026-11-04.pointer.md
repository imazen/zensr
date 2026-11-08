# Routing publication study artifacts

Status: original primary development generation and full artifact audit complete:
235,224 cells and 1,167,767 unique artifacts. The later primary-only fit and
[reserved score comparison](publication_primary_reserved_2026-11-05.md) also
completed; the owner cancelled the reserved pixel-artifact audit. The
[protocol](publication_routing_protocol_2026-11-04.md) owns the scientific design;
the [routing contract](../docs/DEFAULT_ROUTING.md) describes the subsequently
adopted runtime policy. This pointer's directory is the original development
stage, not the complete set of later study directories.

Artifact root verified on i265 on 2026-11-04:

`/home/lilith/data/zensr-nas/output/zensr/publication-routing/2026-11-04`

This is NAS-backed storage. No R2 mirror was created by this study. Generated
images, compressed pixel buffers, metric maps, fitted models and large tables
stay outside git. The content-addressed artifact paths are recorded in each cell.

## Authorized scored-output removal — 2026-11-08

Decoded/repaired RGB buffers and diagnostic maps were removed from all ten
completed runs with the owner's approval. The directory now measures
**7,985,520,640 allocated bytes (7.99 GB)**. JPEGs, reference buffers, score
tables and identities remain. Deletion plans and per-run receipts identify
the removed artifacts; historical audit records moved to `audit-history`.
[Removal counts, retained-input checks and new defaults](scoring_retention_2026-11-08.md).

## Storage footprint before removal

A read-only `du -B1 --max-depth=1` inventory on 2026-11-08 measured
**390,309,404,672 allocated bytes** for this directory; **385,234,767,872 bytes
(98.70%)** belonged to `development-primary`. Original files included JPEGs,
zstd-compressed RGB buffers and compressed Butteraugli maps. These supported
inspection and replay of the experiment; inference does not load this tree.
This is not model-weight storage. Recheck with the same command on the actual
storage directory before treating these dated counts as current usage.

Within `development-primary`, `artifacts` accounts for **383,908,716,544 bytes**,
`cells` for **963,477,504 bytes**, and `scratch` for **338,120,704 bytes**.
These are allocated-byte counts, not sums of logical file lengths.

The compressed primary cell-table archive is **172,706,671 bytes**, excluding
the separately retained coverage and contract files. It preserves measurements,
not the bulk pixel artifacts. Removing pixel artifacts would remove direct
replay/inspection evidence; none was deleted by this inventory.

The frozen harness requested a Butteraugli difference map for both decoded and
restored images and saved every map as compressed little-endian f32. These were
spatial diagnostics for inspecting damage, not router inputs. Analysis reads
the scalar Butteraugli scores from cell tables and includes map paths in its
worst-case records; fitting and runtime do not consume map pixels. Given the
original reference and output RGB buffers plus the exact metric implementation,
maps could be regenerated. After output removal, the decoded/repaired pixels
must first be regenerated from the JPEGs and frozen implementation. Saving every
map was broader diagnostic retention than
the routing analysis needed. The artifact-byte total above combines JPEGs,
RGB buffers and maps; a map-only byte total has not been measured.

## Frozen inputs

| Input | SHA256 |
|---|---|
| `references-audited.tsv` | `8841ec2ae6fff4e9b68d80f97e47e557397e15cdd7e016c1cafc92fba9a791f2` |
| `routing_study` executable | `634e1f7f5b3442ab831756b986bbb1d47f092f8330afa869285e3d3119678c4e` |
| canon4 f16 weights | `c3db9373791270bbe5493d3ee98cd4f62296510fe6931073e2966f433e9a5cf4` |
| Cargo.lock | `87bd7768513c8fd32a5abfafec79b0a2fdc7408ac5a536af5d7806bb8855336b` |

`executable-provenance/record.json` records the preserved executable, compiler,
CPU, lockfile and model metadata. The source tree at `dc5eed79bbc3` contains the
executable's Rust sources. The later analysis changes do not rebuild that binary.
`development-primary/IDENTITY.tsv`, `ENCODERS.tsv` and `FEATURES.tsv` pin the
actual binary, external encoder dependencies and feature definitions. Encoder
identity includes shared-library hashes, not just command names.

The development grid has **235,224 required cells**: 484 references, 61 quality
values, four encoders and 420/444, less the caller-declared jpegli q0 exclusion.
The `COVERAGE.tsv` ledger is authoritative for support and completeness. The
reserved manifest has 291 references and 270 origins and is not yet evaluated.

## Resolve live state

From the repository, with `root` set to the artifact root above:

```sh
just publication-routing-cells status "$root/development-primary"
```

This checks actual cell files and matching live PIDs and writes `PROGRESS.json`.
Run it only when generation has stopped if the directory is accessed over NFS:
a live directory scan was observed blocking worker cell renames. During generation,
verify the local writer PID/scope and staging guard observations instead.
`just publication-routing-storage-count <run-directory>` provides the count when
run on the storage host while generation is stopped; it deliberately makes no
process-health or audit claim.
A stale record or absent PID is not evidence of progress. `COMPLETE` is the
harness's grid-completion marker; `AUDITED.json` is the subsequent hash,
decompression, shape and independent JPEG-read audit. Fitting requires both,
with a cell digest matching the audit. Neither marker is a quality result.

The initial six-worker segment was deliberately interrupted to resume with
twelve workers under the same 16 GiB cap. Its wrapper recorded rc=143, 396 s,
peak RSS 0.52 GiB, minimum available 22,319 MiB and peak load 9.66. It is an
incomplete timing run, not an end-to-end cost measurement. Completed cells were
preserved; the resumed binary validates the same input identities. The event is
recorded in `development-primary/orchestration-events.jsonl`.

## Analysis and reporting

`tools/publication-routing-requirements.txt` pins the tested Python packages.
Set `ZENSR_STUDY_PYTHON` to that environment's Python executable. Run heavy stages
through the resource wrapper, serialized with the sweep:

```sh
just publication-routing-cells audit "$root/development-primary"
just publication-routing-sizes "$root/development-primary" "$root/references-audited.tsv" "$root/development-size" --role router_development
# Run the frozen harness on development-size/references.tsv at the protocol's
# full size/quality/encoder grid into development-size, then audit it.
just publication-routing-cells audit "$root/development-size"
just publication-routing-analysis fit "$root/development-primary" "$root/selection" --size-development "$root/development-size"
```

Fitting checkpoints each grouped fold and stability split. Resuming requires the
same cell digest, study contract, analysis source, threshold source, dependency
requirements and candidate grid. A completed selection cannot be silently refit.
The final `selection.freeze.json` records model hashes and serialized sizes.
The reserved grid must be written under `selection/test-grid`; the Rust harness
refuses that role until its parent contains the selection freeze. Evaluation
also rejects development-origin overlap and changed study identities.

```sh
just publication-routing-cells audit "$root/selection/test-grid"
just publication-routing-analysis evaluate "$root/selection/test-grid" "$root/selection" "$root/reserved-report.json"
just publication-routing-report "$root/reserved-report.json" "$root/report"
```

The renderer writes CSV, Markdown, SVG, PDF and PNG, with source hashes. Synthetic
plot directories are smoke tests only. They are not model-quality evidence.

The owner-approved primary-first amendment uses `fit --primary-only` instead of
the dense fitting recipe above. That fit is complete and frozen in
`primary-fit-20261105-v1` in the canonical array archive; its identity is recorded
in the protocol. The reserved primary score comparison subsequently completed,
with its pixel-artifact audit cancelled. Size-dense development, size generalization, encoder-mode
coverage and measured routing cost remain outstanding for broader claims.
Primary 512-pixel results alone cannot establish full-resolution behavior.

## Artifact writer repair

A deterministic paused-writer test reproduced a concurrent-publication abort in
source matching the frozen executable. The repaired source writes and syncs a
unique staging file, then publishes it through a non-overwriting hard link.
The frozen study executable above was not rebuilt or replaced. Its existing
byte-equality assertion and the mandatory full artifact audit remain in force;
no completed cell or recorded numerical result was changed by this repair.
Debug tests and clippy were serialized with the sweep by temporarily freezing
its own systemd scope, then thawing it. Those interruptions invalidate any use
of sweep wall time as a runtime benchmark.

## Retained encoder dependencies

`encoder-provenance/manifest.json` maps every path in the primary run's
`ENCODERS.tsv` to retained content-addressed bytes. Capture verified every copied
file against its frozen SHA256 before publishing its artifact name. This includes
encoder executables and shared libraries. It is a byte snapshot, not a relocatable
execution environment: paths, dynamic-loader search paths and permissions still
need explicit reconstruction. The capture tool is
`tools/publication_routing_provenance.py` (`just publication-routing-capture`).

## Storage-local audit environment

`audit-runtime-v3/verification.json` identifies the tested offline-built image
`sha256:4d66d2b87c4dc2d6d616fe0c4e3f20073069c25c89e2029eae308f325a4d38a4`.
The context manifest SHA-256 is
`e3ffadc3bb5e13ae2f6d9177ebabc53782252f32b9b5f7a462730c81867f745d`.
Captured dependencies, build/pilot logs and the packager source are retained
alongside it. [Audit purpose and execution](../docs/PUBLICATION_AUDIT.md) describes
concurrency, ownership and resource limits. The original full audit completed successfully;
it was not restarted or replaced by the small pilot.

The completed primary cell digest is
`915744b1966b885ab88c8018d1d480de16f9b6911dd0c5d9804564fbfea389ef`;
coverage digest is
`de7b72e19315fd3caef4cdcba844a95a4018d90c6ebb716ff82e0316e5459f89`.
`development-primary/AUDITED.json` records the tool/dependency identity and
`development-primary/audit.log` retains the resource record.

## Historical dense development execution — 2026-11-04

This section records that stage's configuration, not a current process check.

The supplemented 45-reference, 44-origin grid has 402,408 required cells across
all 20 sizes, the same 61 quality values and four encoders/subsampling pairs.
Its logical output is exposed on i265 at
`/home/lilith/data/zensr-array/development-size` (verified 2026-11-04).
Artifact writes now use owner-approved bounded NVMe staging, with verified
transfers to the canonical array while generation is frozen. Operational guards
trigger at 72 GiB staged or 520 GiB free, inside the approved 80 GiB / 500 GiB
limits; these are polling guards. The original local stage remains preserved.
The metadata and cell tables continue to reside on the canonical array.
The original primary artifact root is unchanged. The complete manifest and
selection metadata are retained beside the dense run. `IDENTITY.tsv` retains
the frozen executable and encoder dependencies. Six workers ran under a 16 GiB
cap. `first-cell-artifacts.json` records successful hash, decompression and JPEG
inspection of the first sampled completed cell; a full audit remains mandatory.

An initial SSHFS transport was replaced by a temporary client-restricted NFS
export of the same direct-array directory. A 15-second syscall profile found
77.04% of aggregate traced syscall wall time in `fsync`; its average traced
`fsync` call took 1,019,737 microseconds. The process was briefly suspended for
the transport switch and resumed with identical files and scientific inputs.
`orchestration-events.jsonl` preserves this event. Elapsed generation time is
not a runtime benchmark. Temporary mount/export cleanup is required after the
study; private execution records retain the exact resolving commands.
