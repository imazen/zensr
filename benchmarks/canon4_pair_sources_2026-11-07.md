# Canon4 pair/source reconstruction — 2026-11-07

**All 48,000 retained HR targets match an imazen-26 source crop exactly.**
All 48,000 encoder/subsampling/quality/clean/generation draws also match seeded
replay. A separate coordinate-based pass verifies every target and both RGB
crop hashes without replaying the RNG. This recovers identities for the
retained dataset; it does not turn the historical policy into the raw canonical
split or establish contemporaneous training-input hashes.

## Actual retained dataset membership

| Canonical bucket | Training pairs | Training sources | Internal validation pairs | Internal validation sources |
|---|---:|---:|---:|---:|
| Train | 39,503 | 910 | 392 | 43 |
| Validate | 4,431 | 91 | 89 | 9 |
| Test | 3,554 | 71 | 31 | 3 |
| Total | 47,488 | 1,072 | 512 | 55 |

There are **1,127 distinct source IDs**, with zero source overlap between
training and internal validation. **7,985 training rows derive from 162 raw
canonical held-out sources.** These count dataset rows, not independent
observations or optimizer draws. The prior policy-only observation was 180
eligible held-out sources; reconstruction identifies actual contributing rows.

## Retained files

On **i265 and dev**, the downloadable bundle is at:

`/mnt/v/output/zensr/canon4-pair-manifest-20261107/`

- `pairs.tsv`: 48,000 array indices, source IDs/paths, crop rectangles, canonical
  and effective buckets, internal partition, codec settings and RGB/source hashes.
- `sources.tsv`: 1,127 source records, actual/manifest hashes and pair counts.
- `sources-with-links.tsv`: the same sources with canonical source/render URLs
  and licence labels preserved.
- `summary.json`, `verification.json`, complete run logs and `SHA256SUMS`.
- `canon4-pair-manifest-20261107.tar.zst`: manifest-only archive, **5,111,726 bytes**.

Archive SHA-256:
`37c9e2e42d22d7422c342201fcd56a925d6b434d994967ef187ef151c6c5fc58`.
Every archive member was checked after decompression, and the mirrored files
and archive were checked by SHA-256. No dataset arrays or weights were changed;
no GitHub release was created. Large manifests remain outside Git.

Small committed records: [reconstruction summary](canon4_pair_sources_2026-11-07.summary.json)
and [independent check](canon4_pair_sources_2026-11-07.verification.json).

## Method and limits

Corpus revision: `187fbf338ce08e8e6654db7f04ddae58d5263da2`. Dataset:
`/home/lilith/tmp/zensr-dejpeg-v4-canon4` on i265. The saved model metadata embeds
the same dataset metadata. Tool launch revision: `f11177a3686a`; coordinate
checker: `5f2a500b4006`. Exact executed-source hashes and dataset input hashes
are in the summary/check records. See [methods and commands](../docs/PAIR_PROVENANCE.md).

Both passes ran with one configured job and a 6 GiB memory cap. Complete logs
are in the bundle. Resource-wrapper completion records:

```text
reconstruction: rc=0 764s | peak-RSS 3.19GiB | min-avail 22430MiB | peak-load 18.30
coordinate check: rc=0 90s | peak-RSS 5.35GiB | min-avail 22543MiB | peak-load 17.40
```

Only **97/1,127 source files** match the older canonical bucket manifest's
byte hash; **1,030 differ**. The table preserves both hashes, and all crop
comparisons refer to the recorded actual source bytes. Differences have not
been classified as metadata-only or independently reconciled with the remote
objects. Canonical membership is joined by source path and ID.

The identity assignment is reconstructed, not an original source log or a
uniqueness proof against pixel-identical sources. Non-clean LR images were not
regenerated with historical encoder binaries. The hash-bound conditioning
arrays were not independently recalculated. Existing weights still inherit
their actual historical training policy.
