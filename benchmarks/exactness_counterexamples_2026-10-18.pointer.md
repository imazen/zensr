# Exactness counterexamples — 2026-10-18

Inputs that showed the shipped x1 pipeline giving different final pixels for
the same JPEG depending on thread count and CPU tier. They are kept outside git
(images are not committed) at, on dev:

`/mnt/v/output/zensr/exactness-2026-10-18/inputs/`

| file | sha256 | bytes |
|---|---|---|
| `girl-576x576-q40-420.jpg` | `0db578fa9ce5b7f49b3f888df8de28059598f086be0bd84e8ca039d8ad24b747` | 20597 |
| `city-576x576-q40-420.jpg` | `e8ffe9536e9aa9d72f2cf7e8623eb414902e59dc6684fd63fccb70fde492bdc3` | 31411 |

Provenance: codec-corpus `gb82/girl-lossless.png`
(`8f0623b15409d6b4b7c1cfa8950b2b608d0bd1a7eb5f382573dd1d7b30b21bcf`) and
`gb82/city-lossless.png` (`92950cec34adafe5a2d8ca5c247ec04df1a8de508f602ef683c838e1e2804aa7`),
576×576. ImageMagick decoded each to raw RGB8 (`magick <png> -depth 8 rgb:<raw>`,
decode only), and zenjpeg git `e277e9c9` encoded it with
`EncoderConfig::ycbcr(40.0, ChromaSubsampling::Quarter)`.

Model: `models/adopted/dejpeg7_graphics/weights_f16.raw` (nf 64, nc 16, x1;
sha256 `7ac1b1f9bf7952336fb27ad0132378fd02b5e77b6b6958ea4ced6612317f0b39`; a local
adoption dump, not tracked in git). `dejpeg_rt24g` (tracked,
`0de929e1bee7bde444d1d8ab46a8fd0d0ff4fef47daf561095d1661008798b80`) showed no
final-pixel difference on these inputs.

Before the fix (runtime source at main `73e5586e`, `restore_jpeg` with
`RestoreConfig::default().with_threads(t)`, final `Restored::to_rgb8`):

- `girl`, auto tier (AVX-512): threads 1/8/12/16 → RGB8 sha256
  `85b6201c467522851d6929c7768c36f40ec30376e85325e783e749bb9b42b973`; threads 2/4
  → `598c902da06406e3e6e9e741cf8977bbd6c9d6e75b497c99233db0f0d614681f`. One byte
  differs: x=296 y=525 R 202 → 203.
- `city`, auto tier → `2673dc8de07a53f66fc9140075f504614a7a0bfd1cd2490bc7fd8c3d078917d4`
  at every thread count; forced v3 →
  `b293bdd06245d3b7dbd99c1366a8d088fb34fcf7b0a5e853e2a1c7f68c5720e8`. One byte
  differs: x=572 y=280 R 107 → 108.

After the fix, every thread count and tier gives one output per input; see
`tools/exactness/product-expect-2026-10-18.json` and
[the repair note](exactness_repair_2026-10-18.md).
