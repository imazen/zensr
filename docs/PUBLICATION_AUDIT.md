# Publication artifact verification

The audit checks complete grid membership, content hashes, decompression, buffer
sizes, and independently decoded JPEG dimensions and subsampling. It does not
recompute quality metrics. Fitting requires its cell digest, coverage digest and
auditor-source identity to match the retained data.

Artifact reads run concurrently (`--workers`, default one); cell-table reads are
serial, with batches capped at 128 cells. Repeated artifacts are verified once,
but every row still checks its dimensions and subsampling. Progress is recorded
atomically in `AUDIT_PROGRESS.json`; only a successful full run publishes
`AUDITED.json`. Previous audit records are retained.

## Run near the storage

Avoid rereading the entire retained pixel/map collection over a slow network
when a suitably provisioned storage-local executor is available. Check its live
work and resource headroom first. Moving an unfinished audit restarts verification;
the in-memory artifact cache is not a resumable checkpoint.

`publication_audit_image.py` creates an offline-buildable Linux x86_64 image
context from the installed Python 3.12 Pillow package and the exact native
`zstd`/`time` binaries, loader and libraries. Every captured file is hashed.
The Python base image must be pinned by digest and already available to an
offline builder. No package installation runs at startup or build time.

```sh
just publication-routing-audit-image "$CONTEXT_ROOT" "$PINNED_PYTHON_BASE"
docker build --network none --pull=false -t "$AUDIT_IMAGE" "$CONTEXT_ROOT/context"
```

Record the resulting image ID and retain the context manifest. Run a completed
pilot first and compare its cell digest and artifact counts with the existing
audit. The image retains the same auditor source; it does not change encoders,
restoration weights, features or metric computation.

On the storage host, run through the resource wrapper and cap the container
itself. Choose CPU assignments from current host availability. Match the data
directory's owner so atomic audit records remain readable by subsequent runs:

```sh
docker run --rm --read-only --network none \
  --user "$(stat -c '%u:%g' "$STUDY_DIRECTORY")" \
  --memory 4g --memory-swap 4g --cpus 4 \
  --cpuset-cpus "$AUDIT_CPUSET" --cpu-shares 128 \
  --mount "type=bind,src=$STUDY_DIRECTORY,dst=/study" \
  "$AUDIT_IMAGE_ID" audit /study --workers 8
```

The entry point applies idle I/O priority and niceness 19, then records GNU
`time -v` output. If the resource wrapper runs around an SSH client, its RSS is
the client's RSS; use the container's measurement for the audit process.

The initial storage-local pilot verified 380 cells / 1,846 artifacts with the
same cell hash as the client-side audits. Eight workers under a four-CPU cap
took 4.86 seconds; GNU time reported maximum RSS 86,744 KiB. These are operational
pilot observations with uncontrolled cache state, not a full-grid speedup claim.
The study's artifact pointer locates the image identity, byte manifest and logs.
