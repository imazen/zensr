# Single-file runtime model preview, 2026-11-08

Implementation review: [PR 1](https://github.com/imazen/zensr/pull/1).
[Format and use](../docs/MODEL_CONTAINER.md).

The binary `.zensr` format stores model layout, color space, portable weights
and the runtime policy. Loading packs weights once for the existing CPU
kernels. This changes packaging and model-policy selection; no inference
speedup or new quality improvement was measured.

## Files attached to the existing model draft

| File | Total bytes | Weight bytes | Routing |
|---|---:|---:|---|
| `canon4.zensr` | 90,110 | 86,406 | Approved decoded default plus coefficient fallback |
| `canon4-coefficients.zensr` | 88,570 | 86,406 | Coefficient-only policy |
| `dejpeg7-graphics.zensr` | 1,191,103 | 1,190,918 | Explicit manual selection required |
| `dejpeg9-gfxycc.zensr` | 1,191,101 | 1,190,918 | Explicit manual selection required |

Canon4's complete file adds 3,704 bytes to the original f16 weights. Weight
bytes are unchanged. Every calibrated policy binds exact metadata and weight
bytes; the packer rejects borrowing canon4 calibration for other weights.
The larger files default to the Quality tier and cannot silently use Auto.

All four files, `CONTAINERS.json`, `CONTAINERS.sha256`, `MODEL-CONTAINER-V1.md`
and `CONTAINERS-README.md` are uploaded to the
[unpublished model draft](https://github.com/imazen/zensr/releases/tag/untagged-4ee8556a90c13af66a9e).
Each of these eight additions was independently compared with GitHub's reported
SHA-256, byte size and uploaded state; the release remained draft/prerelease.
The exact observations and asset IDs are in
[the upload record](model_container_2026-11-08.github.json).
These files are not stored in git. Original model exports and metadata remain
in the draft's existing model-collection archive.

The generating loader revision is `a3fd299dad134a74ed1072647c9bd47355a41c4b`.
A later packer correction uses the existing f32 export filename `weights.raw`;
the four distributed f16 containers are unchanged by that correction.

## Validation

- Default-feature integration suite: 61 passed, zero failures. Five pre-existing
  publication replay/diagnostic tests require explicit external inputs and were
  not invoked. This work added no ignored tests or missing-file early returns.
- No-default-feature integration suite: 44 passed, zero failures. Decoded
  containers fail explicitly without `chooser`; coefficient/manual files load.
- Clippy passes with warnings denied, with and without default features;
  formatting and repository-relative documentation links pass. Doc tests pass
  (there are no crate doc-test examples).
- Six Python packer tests pass: deterministic output, exact weights, metadata
  binding, corruption/truncation, source-constant failure, f32 export convention,
  Span48 layout and refusal to calibrate other weights as canon4.
- Loaded heads reproduce independent sklearn golden predictions and decisions.
  Bundle and raw-model paths produce identical decisions, full reports and
  planar pixels for representative tiny/odd/420/444 inputs, explicit overrides,
  intents and a multi-window input. Altered stored thresholds/gates affect the
  request; non-finite estimates fail rather than silently accepting restoration.
- Synthetic layouts verify f32 loading and color metadata, Compact ×1/×2/×4 and
  Span48 ×2/×4, with exact equality to direct-loaded model output. Separate SR
  models are rejected by the JPEG-restorer conversion.
- The Python-generated canon4 file was loaded by the release-mode repair CLI on
  a 512×512 JPEG with `--routing always --threads 1`; its raw RGB8 output is
  byte-identical to direct loading (`cmp` passed). The larger RGB and YCbCr
  files both completed actual 512×512 CLI repair with explicit Always routing.
  This is an integration smoke check, not a perceptual quality comparison.

Reproduce using `just container-tests`, `just container-tests-no-chooser`,
`just container-check`, `just container-check-no-chooser`,
`just test-model-container`, `just container-doc-tests` and `just docs-check`.
Run builds/tests through the workspace's resource wrapper where required.
CI includes the existing Windows ARM64, macOS ARM64 and cross i686 matrix;
its remote result must be checked on the final PR revision.

## Limits

Quality claims remain those of the existing routing reports, scoped to the
512-pixel comparison cohort. Packaging does not establish optimality, a new
harm bound, full-resolution perceptual quality or calibration for larger models.
The earlier training-data and provenance limitations remain. Smoke RGB8 outputs
are retained locally and were not uploaded; no training pairs or new image
corpus are included in these container additions. The API review is open;
no crate or public release was published.
