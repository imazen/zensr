# Chroma and partial-image boundaries — 2026-11-10

The worst gallery inputs are **32×8**, not 32 pixels high. The source is
neutral grayscale. Three mechanisms contribute; the broad small-bank scores
remain frozen and no runtime routing/projection default changes here.

## JPEG input generation

The fused 420 encoder writes Cb/Cr directly at subsampled resolution but its
partial-bottom padding touched unused full-resolution buffers. A 32×8 input
has four visible chroma rows while its DCT block consumes eight. Missing rows
could be zero or stale. At larger grayscale sizes a preceding full strip can
leave neutral rows and conceal the defect. Width 32 itself is not the trigger.
Independent libjpeg-turbo decoding exactly matches the original native decode.

The original fix `0fee09ca8f7523306da392ac2a2714cd4c783b7e` was extracted
and expanded in [zenjpeg PR 262](https://github.com/imazen/zenjpeg/pull/262),
merged to main as `a9c34a865424c0b33b5b313ce0eca49869c82fd6`.
It pads the actual DCT inputs, including the correct horizontal stride for
pre-subsampled input. A synthetic 32×8 neutral regression fails before the fix.
The partial-geometry, gamma-aware, packed/strided planar controls and library
tests pass. [Encoder report](https://github.com/imazen/zenjpeg/blob/0fee09ca8f7523306da392ac2a2714cd4c783b7e/benchmarks/chroma_edge_padding_2026-11-10.md).
The original bank still uses its frozen older encoder; its bytes are retained.

## Constraint amplification

For the original q40 tiny JPEG, the maximum RGB8 channel spread is **36** after
decode, **38** after guarded canon4, **85/92/105** after half/default/double
projection slack. Mean spread changes from 14.8633 after guards to 35.9023
under default projection. This is mainly coefficient-constraint amplification,
not the raw CNN alone.

`projection_edges` reconstructs the same guard output from retained raw F32
and replays both original default projection pixel hashes exactly. It then
changes ONLY the virtual samples outside the visible frame to the complete
blocks reconstructed from the stored coefficients. Coefficient limits,
decimation/correction, color math and RGB8 rounding are unchanged.

| Original tiny input | Replicated-padding max spread | Encoded-padding max spread | Original A | Encoded-padding A |
|---|---:|---:|---:|---:|
| q40 420 | 92 | 38 | 26.3311 | 66.0495 |
| q50 420 | 94 | 43 | 29.6538 | 65.5447 |

Both encoded-padding results are byte-identical to their guarded CNN outputs.
For q40 Cb, the four hidden decoded row means are 16.997, -11.277, 10.428,
-5.665 JPEG sample units; the last visible row mean is 112.253. Replicating
that visible row supplies a different virtual block and drives visible changes.
This bounded control confirms the mechanism on these two JPEGs. It is not a
validated replacement policy for arbitrary partial blocks, encoders or sampling.
At the time of this diagnostic, runtime projection was unchanged; the runtime correction follows below.

## Learned model chroma and input borders

The GPU experiment uses unchanged author forwards and native-matching weights.
All 18 untransposed default SPANF/DSCF outputs replay frozen bank hashes exactly.
162 controls cover clean/q40/q50 inputs, three sizes, both orientations, three
models and input halos 0/16/32 (replicate, followed by matching crop). All
candidate RGB8 images remain lossless IMIJ entries. No maps were generated.

On clean 32×8 grayscale input:

| Model | Default max channel spread | Default mean spread | Halo-32 max | Halo-32 mean |
|---|---:|---:|---:|---:|
| canon4 raw | 14 | 0.723 | 6 | 0.383 |
| SPANF ×4 | 5 | 0.953 | 6 | 0.591 |
| DSCF ×4 | 8 | 1.293 | 6 | 0.966 |

These RGB networks use zero-padded convolutions and have no exact grayscale
neutrality constraint. Padding controls reduce mean spread here but do not
consistently reduce maxima; larger-input canon4 controls can get worse. No
padding change is adopted. This separates neural chroma from the much larger
inherited JPEG cast and partial-block projection amplification.

Artifacts on i265: `/mnt/v/output/zensr/sr-bank-20261110/`:
`chroma-edges-v2/` (complete; failed v1 preserved), `witness-stages-v1/`, and
`projection-edges-v1/`. Result records bind source hashes; original rows/images
are unchanged. The projection control is implemented through the existing
public projection kernels on explicitly extended planes. Its analytic DC test
also covers unclipped hidden samples outside the display range.

```text
GPU controls: rc=0 3s | peak-RSS 1.34GiB | min-avail 22702MiB | peak-load 0.77
projection:   rc=0 0s | peak-RSS 0.01GiB | min-avail 23596MiB | peak-load 0.48
```

No speed extrapolation, full-resolution quality claim or production selector
follows from these small witnesses. The subsequent shared-runtime correction and its 16,104-case replay are
reported in [the qualification](projection_fix_2026-11-10.md). Regenerated
input banks must get a new identity rather than overwrite this study.

Retained mirror (SSH streaming, no NFS):
`tower:/mnt/disk1/coefficient/output/zensr/sr-bank-20261110-edges-v1.tar`,
16,670,720 bytes, SHA-256
`675ab13d576590f38fa1163d8f9e3c66dcb08778c84b29830405fc2d2290997d`.
Sender/receiver hashes match, file/directory fsync completed, and three seeded
random candidate members read back with matching hashes. `EDGE_STUDY.json`,
`EDGE_BACKUP_RECEIPT.json` and `EDGE_READBACK.json` preserve the identities.
Final-source projection output replays the first control results exactly.
The live gallery's native-LR expansion now separates guarded CNN output from
coefficient projection and links them through the existing flicker viewer.

## Runtime correction following the owner’s request

The shared `zensr-micro::consist::project_plane` now reconstructs the hidden
part of partial blocks from their dequantized coefficients. It preserves
out-of-range hidden IDCT values in JPEG units, overlays only visible candidate
samples, and applies the existing coefficient limits. Fully visible blocks
retain their existing arithmetic. Completely invisible MCU blocks are ignored.
No new public signatures or looser bounds are introduced.

A decoded partial-block invariance regression fails before the fix and passes
afterward for both natural and zigzag coefficient order, every 1–8 pixel
width/height and three quantizer steps. Further controls exercise unclipped
hidden samples, mixed partial geometries and error to an in-box decoded target.
Runtime quality/integration measurements are distinct from the frozen bank above.
