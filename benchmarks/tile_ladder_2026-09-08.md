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

A floor on the tile stops subdivision running to nothing — the loop ends only
when there are enough tiles, and on a 64px image at 12 threads that would drive
the tile to zero.

**The floor wants to be much lower than the halo cost alone suggests, and
getting this wrong made the whole change invisible.** The first version used
`6 x halo`, reasoning that `(1 + 2*halo/tile)²` makes `2*halo` cost 4x in
wasted border. Measured, paying that 4x is still right, because it buys threads:

| | best tile | ms | what a 6x floor forces | ms |
|---|---|---|---|---|
| halo 10, 128px, 12T | 44 (4.4x halo) | 3.8 | 76 | 6.6 |
| halo 10, 128px, 28T | 44 | 3.8 | 140 | 12.3 |
| halo 18, 128px, 12T | 44 (2.4x halo) | 70.2 | 140 | 156.0 |
| halo 21, 256px, 12T | 86 (4.1x halo) | 154.0 | 134 | 204.4 |

A `6 x halo` floor is 74% slow at 128px on the realtime model and 2.2x slow on
the quality one. `2 x halo` lands within 8% of the measured optimum in every
cell swept, across halos 10/18/21 and 8/12/28 threads. With the 6x floor the
change was **invisible end to end** (median −0.5% through `prod_bench`), because
the production models' halos are large enough that it blocked every
subdivision — the floor, not the rule, was doing the deciding.

## Result

The rule changes 47 of the swept cells. Both tiles measured **inside one
binary**, interleaved, 3 paired reps, across the three shipped models
(halo 10 / 18 / 21) at 128-768px and 4/8/12/28 threads:

**46 of 47 cells win. Median +51.1%, best +75.0%, worst −7.1%.**

| model | size | threads | ladder | new | faster by |
|---|---|---|---|---|---|
| realtime (h10) | 128 | 28 | 140 | 44 | **+70.2%** |
| realtime | 192 | 8 | 268 | 76 | **+71.6%** |
| realtime | 256 | 8 | 268 | 92 | **+74.0%** |
| realtime | 256 | 28 | 140 | 44 | +55.2% |
| realtime | 512 | 4 | 396 | 268 | +51.3% |
| realtime | 512 | 28 | 140 | 92 | +30.8% |
| realtime | 768x512 | 8 | 268 | 204 | +28.0% |
| quality (h18) | 128 | 28 | 140 | 44 | **+59.5%** |
| quality | 256 | 8 | 268 | 92 | **+70.9%** |
| quality | 512 | 4 | 396 | 268 | +47.9% |
| SR span (h21) | 192 | 8 | 262 | 70 | **+67.3%** |
| SR span | 256 | 8 | 262 | 86 | **+68.1%** |
| SR span | 512 | 4 | 390 | 262 | +50.7% |
| SR span | 512 | 28 | 134 | 86 | **−7.1%** |

Full table: `tile_ladder_ab_2026-09-08.tsv`. The single loss is the one cell
where the starvation fix does not pay for itself — 16 tiles across 28 threads is
genuine starvation, but at halo 21 the 2.2x border cost of the smaller tile
outweighs the twelve extra threads.

Tiling stays **bit-exact in the tile size** — checksums are identical across the
whole sweep on all three models — so this changes speed only.

## End to end

`prod_bench` (restore_jpeg -> guarded x1 model -> S10 projection -> RGB, then the
chained x2 SR step), 3 interleaved paired reps, raw log
`tile_ladder_e2e_2026-09-08.log`:

| stage | 256px, 12 threads | every other cell |
|---|---|---|
| restore | 233.9 -> 135.6 ms, **+41.5%**, 3/3 | flat, ±2% |
| sr_x2 | 198.4 -> 126.1 ms, **+36.5%**, 3/3 | flat, ±2% |
| chain | 431.3 -> 261.7 ms, **+39.2%**, 3/3 | flat, ±2% |

That is exactly right, and the flat cells are the point: 256px at 12 threads is
the **only** shape in this harness where the rule changes the tiling. At 64px
the image is one tile either way; at 256px with one thread the ladder is not
starved; at 1024px and above the ladder already produces plenty of tiles and no
runt. Everything else is unchanged **by construction**, and reads flat, which is
the control this comparison needed.

The gain is not confined to that one shape — `tile_probe` finds it across
128-768px at 4/8/12/28 threads on all three models. It is confined to the shapes
`prod_bench` happens to sweep.

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
