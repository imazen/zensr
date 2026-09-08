# Methodology for shipping binaries to end users

zensr ships as **binaries to end users**, not as a service. That single fact
invalidates most of the usual safety net, and this document exists because the
gates have to move accordingly.

## 1. What we do not have

| Service deployment assumes | We have instead |
|---|---|
| Percentage rollout | Every user gets the same build at once |
| Kill switch / instant rollback | A binary already on a user's disk |
| Telemetry to notice a regression | Nothing, unless the user reports it |
| One CPU you chose | Every x86 and ARM the user happens to own |
| One input distribution you observe | Arbitrary, including hostile, files |
| Hotfix in minutes | A release cycle |

So the gate is **before the build**, and it has to be a proof rather than a
plan to watch. Three consequences run through everything below: a defect ships
to 100% of users simultaneously; the CPU picks the code path, not us; and a
wrong number in a doc is a wrong number in someone's product decision.

## 2. Measurement rules

**A benchmark must name the arm it actually measured.** Not the arm it intended
to. `benches/kernel_tiers.rs` hardcoded `"v3(avx2)"` on all x86_64 while the
dispatch ladder picked `v4x` — every x86 reading of that bench named the wrong
tier until 2026-09-08. Tier labels are now probed at runtime. Anything that
selects a code path at runtime (SIMD tier, model, policy, encoder family) must
be *reported from the runtime*, never from a constant that a human kept in sync.

**A baseline must be verified to be the baseline.** The same bench disabled only
`X64V3Token` while claiming a "scalar" arm. Because V4x is a superset, a ladder
change could have left AVX-512 live in both arms — the ratio would read ~1.00x
and pass as "no sub-parity kernel" while measuring nothing at all. A control
that can silently become the treatment is not a control.

**Paired statistics, always.** Median of per-file differences plus win fraction;
never the difference of two per-method medians. A model that wins on the cell
mean can still lose on most individual files, and has here.

**Report the intercept.** Fit `total = α + β·MP` and give both. A ms/MP number
alone is meaningless at thumbnail sizes, where α dominates — measured for the
realtime tier at 12 threads: `6.1 ms + 192.0 ms·MP`.

**Split by reference provenance,** and say when two populations are not a
controlled comparison. In the canonical corpus PNG/JPEG references are nearly
collinear with graphic/photographic content, so that split mixes two effects;
`tools/rescore_report.py` carries the warning where the number is read.

## 3. Gates that must pass before a build ships

Ordered by what a failure costs a user.

1. **Bit-identical output across every SIMD tier, on real hardware.** A user's
   CPU picks the tier. If v4x and v3 disagree by one ULP, two users get
   different pixels from the same input, and any cached derivative diverges.
   Equivalence is already tested (`simd_matches_scalar_reference`,
   `arbitrary_dims_simd_vs_scalar`, `wino_dispatch_matches_direct`,
   `tiled_matches_whole_image`, `px_strided_matches_tight_and_raw`) and CI runs
   6 platforms plus i686 via cross. **What is missing is a per-tier golden**:
   the same input hashed through v4x, v3, neon and scalar, compared to a
   committed digest. Tests prove the kernels agree with each other; a golden
   proves this build agrees with the last one.
2. **Resource bounds on untrusted input.** `restore_jpeg` has a *time* budget and
   no pixel or memory ceiling. On a server that is a scaling problem; on a user's
   laptop it is the application dying. `prod_bench` already has the RSS harness
   (`ZENSR_PB_ONE=<side>` under `/usr/bin/time -v`) — run it across the size
   ladder, commit the numbers, then bound the API by them. No estimate: heaptrack
   or `/usr/bin/time -v`, per the workspace rule.
3. **Fuzzing.** There is no `fuzz/` directory. Every sibling codec repo has one
   and a nightly farm. zensr's projection path consumes attacker-controlled
   quantization tables and coefficients, and the library crates carry ~63
   `unwrap`/`expect`/`panic!` sites. In a service a panic is a 500; in a shipped
   binary it is a crash in someone's product with no telemetry and no rollback.
4. **The identity gate, re-derived.** Ungated, the model *lost* up to 2.1 ssim2
   and harmed 91% of files. It is the one constant whose failure mode is
   "silently degrades images that were already fine", which is exactly the defect
   an end user cannot detect and we cannot observe. It is currently fitted on a
   corpus that no longer exists.
5. **Reproducible provenance for what ships.** 4 of 47 checkpoints record the
   commit they were trained at; 3 embed their dataset. A binary cannot be
   hotfixed, so "which model is in the build, and what produced it" has to be
   answerable from the build itself.

## 4. What replaces staged rollout

Since exposure cannot be ramped, ramp the **input** instead:

- **Content-class reporting, never an aggregate.** The dejpeg9 route gains
  +2.72 ssim2 on graphics and *loses* on photographs from q55 up. An aggregate
  hides that; a per-class table cannot. Every ladder is reported per content
  label.
- **Conservative defaults with an escape.** The router already passes high-q
  images through unchanged. Default to the narrowest configuration that is
  measured to help, and give the caller an explicit opt-in for the rest —
  a route, not a default (`dejpeg9_gfxycc` is exactly this shape).
- **A no-op path that is genuinely no-op.** When the gate declines to restore,
  the output must be the plain decode, bit-identical. That is testable and
  should be a golden.
- **Version the output identity.** Any downstream cache must invalidate when the
  model or the constants change, which means the version has to be part of what
  the caller can see.

## 5. Standing traps, measured in this repo

Each has cost real time here; none is hypothetical.

- **n=64 produces false positives.** Two shipped-looking results died on
  re-testing. Use the 20-random-split protocol for anything intended to ship.
- **The metric floor is ~0.3 ssim2.** Differences below it are not resolvable.
- **`pgrep -f` matches your own shell.** Use `pgrep -x`, a recorded PID, or a
  completion marker.
- **Check for existing data before generating any.** Three runs were launched and
  killed here after the answer turned out to already exist.
- **A perceptual fingerprint is a candidate generator, not a verdict.** The 16x16
  luma test over-flags document scans; confirm against identity where the corpus
  has one.
- **`ln -sf` overwrites but never removes.** A corpus builder that does not clear
  its output keeps files a later filter excluded — six training images survived
  into an audited eval corpus that way.
