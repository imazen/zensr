# Quality-model precision experiment retention

Verified 2026-11-08 on i265:
`/mnt/v/output/zensr/quality-tract-int8-20261108/`.

The storage copy is `tower:/mnt/disk1/coefficient/output/zensr/quality-tract-int8-20261108/experiment-v2.tar`.
It contains **1,529,344,000 bytes**, SHA-256
`4c0c3aef104f19587b9863e897d7a774fb87549964913ddeb0906dac43574fc8`.
Sender/server byte counts and whole-stream digests match. The server synchronized
both the file and its parent directory. Three seeded-random archive members and
three critical members were read back and matched source hashes: baseline f16
weights, full quality results, and the uploaded model bundle are among them.

The archive retains both hosts' raw timing/round records, profiler captures,
frozen executables and source snapshots, complete quality cells/table, retained
inputs, calibration/exports and the verified model package. Cargo build caches
and installed Python environments are excluded; experiment executables remain
outside those caches and are included. All 2,960 required inputs and all five
three-process timing blocks are represented.

An NFS stale handle stopped the initial mount-based attempt before writing.
The first direct-stream attempt omitted a root-owned profiler file and was
rejected by the tar exit-status gate. That incomplete attempt is preserved as
`experiment-v1.incomplete.tar`. Correcting access to the owned capture allowed
the complete v2 archive; no measured outputs or study inputs changed.

[Measurements and limits](quality_int8_2026-11-08.md) ·
[Verified draft assets](quality_int8_assets_2026-11-08.json).
