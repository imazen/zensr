# Where sweep time actually goes — 2026-08-04

Prompted by a good question: the sweep shells out to `cjpeg` per cell, so why not
encode in-process with zenjpeg in sequential mode, multi-stream?

**The encoder finding is real and large — and the sweep is not encode-bound, so
it does not help this run.** Both halves measured rather than assumed.

## In-process zenjpeg vs the cjpeg subprocess

`encode_bench` bin, 512² crops, same geometry the sweep uses.

Single-threaded:

| config | enc/s | ms/enc | output |
|---|---|---|---|
| `cjpeg` subprocess (what the sweep does) | 197.6 | 5.06 | 19,718 KB |
| zenjpeg sequential, no Huffman opt | 501.6 | 1.99 | 18,505 KB |
| **zenjpeg sequential, Huffman opt** | **507.2** | **1.97** | **16,943 KB** |
| zenjpeg progressive (the config default) | 313.7 | 3.19 | 16,300 KB |

Multi-stream (one image per worker):

| workers | enc/s | vs subprocess |
|---|---|---|
| 1 | 507 | 2.6× |
| 4 | 1,762 | 9× |
| 8 | 3,451 | 17× |
| 16 | 5,566 | 28× |
| 24 | **7,183** | **30×** |

Three things worth keeping:

- **Sequential is 1.6× faster than progressive**, and `EncoderConfig::ycbcr`
  defaults to *progressive* even though `ProgressiveScanMode`'s own `#[default]`
  is `Baseline`. Sequential has to be asked for explicitly.
- **Huffman optimization is free** — 507.2 vs 501.6 enc/s — and produces 8%
  smaller files. There is no reason to turn it off.
- Scaling is near-linear to 24 workers.

## But the sweep is not encode-bound

The sweep runs at 11.2 arm-rows/sec on 6 threads = 3.73 cells/sec, so it needs
**3.73 encodes/sec against a 197/sec floor — 1.9% of capacity.** Even the slow
path has 50× the headroom needed.

Per-call cost at 512² (`metric_cost` bin), which is where the time is:

| | ms/call |
|---|---|
| ssim2 | 19.75 |
| butteraugli n3 | 16.58 |
| zenjpeg encode | 1.95 |
| psnr | 0.00 |

A whole encode costs a tenth of one ssim2 call, and there is one encode per cell
against three arm-rows of metrics.

## The optimization that follows, and its real size

Butteraugli is 46% of *metric* cost and **no routing curve zensr ships uses it** —
they are all fitted on ssim2. So `ZENSR_EVAL_NO_BUTTER=1` now skips it and
reports NaN (never a plausible number, so a skipped column cannot be mistaken
for a measured one).

Measured A/B on an identical slice: **27.2s → 23.4s, 14%.** Not the 46% the
per-call numbers suggest — model inference and decode dominate more than the
metrics do. Predicting 46% from a component share would have been wrong; the A/B
is why it is 14% here.

**Left off by default and not applied to the runs in flight.** Butteraugli
disagreeing with ssim2 is itself a finding — that is the whole `renders` result
(§0.3) — and dropping it from one encoder would make the XL dataset asymmetric
immediately after adding a rule to report all three metrics paired. 14% of one
run is not worth that.

## What to use it for

Future sweeps should encode in-process: 30× on the encoder costs nothing to
adopt, removes ~53k process spawns per encoder from a full XL grid, and matters
much more on boxes where `fork` is expensive or where the grid is encode-heavy
rather than metric-heavy. The runs currently in flight stay as they are —
restarting them costs more than the 3% the swap would return.
