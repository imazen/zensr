# Routing publication study artifacts

Status: development sweep in progress; no new measured routing frontier yet.
The [protocol](publication_routing_protocol_2026-11-04.md) owns the scientific
design. Runtime defaults remain unchanged; deployment selection follows review.

Artifact root verified on i265 on 2026-11-04:

`/home/lilith/data/zensr-nas/output/zensr/publication-routing/2026-11-04`

This is NAS-backed storage. No R2 mirror was created by this study. Generated
images, compressed pixel buffers, metric maps, fitted models and large tables
stay outside git. The content-addressed artifact paths are recorded in each cell.

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
just publication-routing-analysis fit "$root/development-primary" "$root/selection"
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

Still required: completed primary sweep and audit, frozen fitting, reserved
evaluation, dense size generalization, encoder-mode coverage, and measured
routing cost. Primary 512-pixel results alone cannot close the publication gates.
