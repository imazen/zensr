# The tile ladder picked a tile SIZE. It should pick a tile COUNT.

2026-09-08. Host `dev` (WSL2 on a **7950X**). `cargo build --release`, no
`-C target-cpu=native`. Tool: `zensr-bench/src/bin/tile_probe`. Raw data:
`tile_ladder_2026-09-08.tsv` (the full 7-tile sweep) and
`tile_ladder_ab_2026-09-08.tsv` (the paired rule comparison).

Re-measured after the conv3x3 tap-load change, per the standing rule that a
kernel change invalidates the tile tuning. It did — but what it exposed was not
drift. It was a defect the ladder had from the start, and only a size sweep
could see it.

## The defect

`upscale_tiled`'s default picked a tile **size** from the thread count: 512 at
1-2 threads, 384 at 3-4, 256 at 5-8, 128 above. A size that does not divide the
image leaves a **runt** tile, and a tiled run is paced by its **largest** tile.

At 512px the 3-4 thread rung picks 396, so the image splits into 396 and 116.
The predicted cost is the ratio of convolved areas, `(416/288)² = 2.09x`. The
measured cost is **2.25x**. At 256px the same rung gives ONE tile for the whole
image, so three of four threads idle.

Neither shows up at 1024px, which is where the ladder was originally measured.

## The fix

Pick the tile **count** and let the size follow, so the tiles are equal by
construction — but only where the ladder's own tiling is pathological:

- **starved** — fewer tiles than threads, so threads sit idle;
- **lopsided** — the last tile is under half the others, *and* there are few
  enough tiles (`<= 2x threads`) for one of them to pace the run, *and* there is
  more than one thread.

Otherwise the ladder's tile is kept. Both conditions are needed, and so are
their guards:

- Re-tiling unconditionally costs **6.7%** (realtime) and **11.9%** (quality) at
  1024px/4 threads, where the ladder's 396 and the even 348 give the same tile
  count and the same total convolved area, and 348 is simply a worse tile. The
  tile-size landscape is bumpy in ways an area model does not predict.
- Re-tiling for lopsidedness at **one thread** costs **15.2%** at 768x512: with
  one thread the tiles run sequentially, so imbalance is irrelevant and the
  extra tiles are pure added halo.

A floor of `6 x halo` stops subdivision before the discarded border dominates —
halo cost is `(1 + 2*halo/tile)²`, so `6*halo` pays 1.78x and `2*halo` pays 4x.
Without it a 128px image at 12 threads is split into four 44px tiles, each
paying 2.1x, chasing parallelism it has no work for.

## Result

The rule changes 16 of the measured cells. Both tiles measured **inside one
binary**, interleaved, 3 paired reps:

| model | size | threads | ladder | new | faster by | wins |
|---|---|---|---|---|---|---|
| realtime | 128x128 | 4 | 396 | 76 | **+51.4%** | 3/3 |
| realtime | 128x128 | 8 | 268 | 76 | **+50.4%** | 3/3 |
| realtime | 128x128 | 12 | 140 | 76 | **+50.7%** | 3/3 |
| realtime | 256x256 | 4 | 396 | 140 | **+64.9%** | 3/3 |
| realtime | 256x256 | 8 | 268 | 92 | **+72.7%** | 3/3 |
| realtime | 256x256 | 12 | 140 | 76 | +24.8% | 3/3 |
| realtime | 512x512 | 4 | 396 | 268 | **+51.7%** | 3/3 |
| realtime | 512x512 | 8 | 268 | 172 | +26.8% | 3/3 |
| realtime | 768x512 | 8 | 268 | 204 | +13.4% | 3/3 |
| quality | 128x128 | 4 | 396 | 140 | +1.7% | 3/3 |
| quality | 128x128 | 8 | 268 | 140 | +1.9% | 3/3 |
| quality | 256x256 | 4 | 396 | 140 | **+61.0%** | 3/3 |
| quality | 256x256 | 8 | 268 | 140 | **+59.4%** | 3/3 |
| quality | 512x512 | 4 | 396 | 268 | **+43.8%** | 3/3 |
| quality | 512x512 | 8 | 268 | 172 | +20.3% | 3/3 |
| quality | 768x512 | 8 | 268 | 204 | −3.0% | 0/3 |

15 wins, 1 loss. The loss is the one cell where the starvation fix does not pay
for itself on the quality model; the same cell is +13.4% on the realtime one, so
the mechanism (6 tiles across 8 threads) is right even where the trade is not.
Every cell the rule no longer touches keeps its old behaviour by construction.

Tiling stays **bit-exact in the tile size** — checksums are identical across the
whole sweep on both models — so this changes speed only.

## Comparing two BUILDS of the rule does not work

The first attempt at this A/B built one binary per rule and compared their
defaults. It reported a −11% "regression" at 1024px/12 threads where **both
rules choose the identical tile**. Measuring the same explicit tile 140 in each
binary: 109.2 ms against 115.9 ms. Editing the rule moves code layout and
inlining, and that bias — about 6% here — lands on every delta.

The sound comparison is both rules' chosen tiles measured **inside one binary**,
which is what `ZENSR_TP_TILES=a,b` is for. Every number above is from that.

## Rules fitted and rejected

Scored against the 7-tile sweep (20 cells), penalty against the per-cell best:

| rule | median | mean | worst |
|---|---|---|---|
| shipped (size from thread count) | +3.6% | +20.4% | +125.2% |
| even tiles | +2.0% | +9.4% | +84.5% |
| even + one tile per thread | +2.0% | +6.8% | +46.9% |
| even + `sqrt(1.5T)` tiles | +2.0% | +7.2% | +29.8% |
| cost model over the tile count | +0.0% | +4.4% | +25.7% |

The cost model — minimise total convolved area `(side + 2*halo*n)²` divided by
wave efficiency `n²/(ceil(n²/T)*T)` — scores best and is the only one that
splits the two models correctly at the same thread count, since the halo-18
model wants fewer, bigger tiles than the halo-10 one and a thread-count ladder
cannot express that. It was **not** shipped: it over-sizes at 12 threads (up to
+25.7%) by an amount no term in it explains, and it proposes tiles far outside
the measured range at low thread counts. Adding a working-set cap to fix the
12-thread behaviour made every score **worse** at every budget from 8 to 96 MB,
so the cache hypothesis for that residual is falsified, not merely unproven.

The residual is the tile count. It is worth up to ~25% and needs a model that
explains the 12-thread behaviour before it can ship.
