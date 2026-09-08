# What the existing trained models are worth

Audited 2026-09-08, after the corpus repoint (`CORPUS-REPOINT-IMPACT.md`).
47 checkpoints under `models/adopted/`. Short answer: **all of them are
corpus-provisional, most have no recoverable provenance, and the README's
quality-tier default sits in a lineage that trained on corrupt ground truth.**

## 1. Provenance barely exists

| | count |
|---|---|
| checkpoints with a `meta.json` | 47 |
| …recording the **git commit** they were trained at | **4** |
| …embedding the **dataset's own meta.json** | **3** |

The shipped realtime tier has neither. Worse, its generated `repro.sh` says
*"dataset /home/zen/zensr-d7f16 (its meta.json embedded in model meta)"* while
`meta.json` has `"dataset_meta": null`. A generated file asserting a provenance
link it did not write is worse than no link, because it stops the next person
looking.

Training datasets live in two places that are not the repo and not durable:
`/home/lilith/tmp/…` (scratch — and demonstrably subject to the local ext4
zero-hole corruption, see §3) and `/home/zen/…` (a fleet node). The shipped
model's dataset, `/home/zen/zensr-d7f16`, **is not on this machine at all**, so
its training set cannot be checked for overlap with the new held-out split
without reaching that node.

## 2. Every model predates the repoint, so every number is provisional

All 47 were trained before 2026-09-08, which means through `/mnt/v/imazen-26`
(the pre-curation acquisition corpus, since deleted) and the
"first-8-sorted ∪ 64 pinned" exclusion scheme, which was never a split and leaked
twice.

The README's headline table compounds this. Its stated basis is *"clean PNG
references (`/mnt/v/imazen-26-clean`), pinned eval split, n=64/cell"*:

- `/mnt/v/imazen-26-clean` was derived from the deleted invalid root and is now an
  empty shell (74 files).
- the pinned eval split was removed in the repoint as invalid — its directory keys
  named a corpus that no longer exists.
- under the canonical split those 64 files are now distributed across train,
  validate and test arbitrarily, so some of them are training data today.

**No number in that table is currently reproducible or verifiable.** They are not
necessarily wrong; they are unfalsifiable, which for a public README is the same
problem.

## 3. The corrupt-ground-truth bug reached the shipped quality tier

`~/tmp/zensr-dejpeg-v2/off/hr_u8.npy` has **20 fully-zeroed ground-truth crops**
(indices 4587–4606, plus 4607 partially zeroed) out of 24,000 — 0.083%. A ~1 MiB
contiguous run, which is the signature of the open local ext4 zero-hole bug, not
a pipeline defect. `zensr-dejpeg-v3/off` is clean: **0 zeroed crops**.

It is not a harmless no-op. For all 20, the *input* is a real JPEG crop
(`lr` mean 128–237, std 27–60) while the target is pure black — so those pairs
actively teach "given this JPEG, output black", rather than being skipped.

Three models trained directly on it, per their training logs
(`~/tmp/zensr-dejpeg{2b,3-off,3-auto}-train.log` all reference the v2 dataset):
`dejpeg2b_off`, `dejpeg3_off`, `dejpeg3_auto`. The fine-tune chains carry it:

```
dejpeg7_graphics <- dejpeg4_policy <- dejpeg3_off <- dejpeg2b_off <- dejpeg2_off <- dejpeg_1x
dejpeg9_gfxycc   <- dejpeg4_policy <- dejpeg3_off <- ...
dejpeg_rt24g     <- (init=scratch, dataset /home/zen/zensr-d7f16)
```

| README row | model | corrupt-v2 lineage |
|---|---|---|
| quality (default) | `dejpeg7_graphics` | **YES** |
| low-q graphics route | `dejpeg9_gfxycc` | **YES** |
| realtime (committed weights) | `dejpeg_rt24g` | no — trained from scratch on another dataset |

So the **committed, publicly downloadable weights are clean of this**, but the
**default quality tier is not**. At 0.083% of pairs the practical effect is
probably small — but "probably small" is a guess, and it is trivially avoidable:
v3 is intact, so any retrain simply uses good data.

## 4. What to do, in order

1. ~~**Do not retrain first.**~~ **DONE 2026-09-08 —
   `benchmarks/rescore_canonical_2026-09-08.md`.** All three shipped models
   re-scored on the canonical held-out split, 0 training files scored, 0 files
   skipped. Both tiers survive: lower at q15 (−27%), *higher* at q35–q90 than
   published. The dejpeg9 graphics route is confirmed at 1.7–2.7× its claimed
   size and is negative on photographs from q55 up, so it must stay a route.
   The published numbers were unverifiable, not wrong.
2. ~~**Correct the README before it is quoted again.**~~ **DONE** — the table now
   carries measured numbers with the held-out basis stated.
3. **Retrain the quality tier from the repointed pipeline**, which uses v3-clean
   data by construction and the canonical split. `dejpeg7_graphics` is the one
   that matters — it is the default. The re-score justifies this on the corpus
   swap and the corrupt lineage, **not** on a quality failure: the model works.
   The one open question it raised is the **q15 gap** — both models lose ~27% of
   their published low-q gain on the new PNG population, consistently. Low-q is
   where web traffic and the routing constants live, so understand it first.
4. **Fix the provenance hole at the source**: record `git commit`, embed the
   dataset `meta.json`, and refuse to write `repro.sh` claims that are not true.
   A checkpoint whose training set cannot be identified cannot be audited for
   leakage, which is the whole reason this audit was needed.
