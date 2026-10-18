# Compact inference Coz probe — 2026-10-18 UTC

No runtime optimization or speedup is claimed. This verifies profiler mechanics
and repeatable output for one compact model and shape. Other models, larger
sizes, tiled/multithread execution, and forced SIMD tiers remain unprofiled.

Runtime source: `12e1054d`. Harness: `4abe56e7`.
Machine: `dev`, AMD Ryzen 9 9950X3D. Rust 1.99.0; Coz 0.2.5;
coz-rs 0.1.3. No `target-cpu=native`.

Artifacts checked on `dev`: `/mnt/v/output/zensr/coz-2026-10-18/`.
`MANIFEST.json` records model, source, shape, and per-file SHA256 hashes.
`probe.jsonl` SHA256:
`ef45de4058ca1709ec93d3791f7deffeb809ca48c92fcc90a6c5c21fe29cb652`.
R2 and tower mirrors have not been created.

Run `just coz-build`, then:

```sh
just coz-run probe.jsonl models/adopted/dejpeg_rt24g/weights_f16.raw 24 8 1 64 64 500 0
just coz-repro <new-output-directory>
```

The probe recorded four source-attributed experiments with selected samples
and 484 completed progress events during its measurement windows. This is
insufficient to rank optimization candidates. The two 49,152-byte raw outputs
from `coz-repro` compare equal with `cmp`; both have SHA256
`1d58a4684e35d791263aeab8f8f55395d46ee783b274e0217e77b754ad015232`.
This verifies repeatability on this tier and input, not cross-tier parity.

Resource wrapper output for the probe:

```text
run-heavy: done rc=0 3s | peak-RSS 0.03GiB | min-avail 21892MiB | peak-load 26.46
```

The whole-workspace baseline clippy command failed dependency resolution:
the workspace pins archmage 0.9.29, while the local zenjpeg override requires
0.9.30. The standalone tool retains the runtime's existing SIMD pin and
passes formatting and clippy. Cargo emits an unused-local-patch warning for
that unrelated zenjpeg override. Existing runtime unit tests have not passed
through this isolated workspace: Cargo rejects testing a dependency with
dev-dependencies when it is not a workspace member. Preserve those tests and
resolve the workspace build before landing any production runtime change.

Harness details: [tools/coz-profile/README.md](../tools/coz-profile/README.md).


## Follow-up: existing unit-test gate

`14435363` adds an isolated `runtime_unit` target pointing to the actual runtime
crate root. It executes the existing tests with unchanged assertions, without
resolving the benchmark codec graph. `just coz-test` passed with 19 tests;
`--no-default-features` passed with 18; `--features tier_v4` passed with 19.
The full workspace graph and optional `px` integration remain outside this
local gate. Logs are `runtime-matrix.log` and
`clippy-runtime-unit-final.log` in the artifact directory above.
The runtime source remains unchanged.
