# eval_split — where the split actually lives

**Not here.** The canonical imazen-26 train/validate/test split is part of the
corpus repository, `github.com/imazen/imazen-26`, at
`manifests/{split_map,train,validate,test}.tsv`. The rule is the last digit of
the 4-digit image id — {0,2,4,6,8} train, {1,3,5} validate, {7,9} test —
identical to zenmetrics `scripts/picker/origin_split.py`. Totals: 1,084 / 658 /
418 over 2,160 images.

Python reads those canonical buckets through `tools/corpus_split.py`, then
applies zensr's related-image grouping. Rust's
`zensr_bench::canonical_holdout()` reads the generated **effective** split at
`eval_split/imazen26_effective_split.tsv`, not the raw corpus buckets. Point
`IMAZEN26_REPO` at the corpus checkout if it is not at `~/work/zen/imazen-26`.

**Why nothing is vendored here.** zensr kept its own eval list twice — a
hand-maintained 64-file pin (`imazen26_eval_files.tsv`, keyed to the deleted
`/mnt/v/imazen-26` layout), and then a re-derived split of its own. Both drifted
from the corpus, and the first leaked training images into an eval twice. A copy
of someone else's split is a copy that will be wrong later.

`tools/corpus_split.py` same-buckets related images using the lowest-ID member's
bucket. Its grouping includes patent-page variants, brochure pages, screenshot
sites and hurricane advisories; the helper documents the limitations for books.
At corpus revision `187fbf338ce08e8e6654db7f04ddae58d5263da2`, checked on
2026-11-07, **309 memberships change**. Effective counts are **1,151 train /
648 validate / 361 test**. In particular, 103 canonical validation and 77
canonical test images enter the effective training pool; 113 canonical training
images leave it. This does not preserve the raw canonical held-out boundary.
Both training and evaluation must use the same effective policy when assessing
existing models. See [model training and provenance](../docs/MODELS.md).

## What IS here

| file | what |
|---|---|
| `imazen26_effective_split.tsv` | Generated local contract; run `just split`; not committed |
| `picker_safe_origins_2026-09-08.txt` | **current** — 171 leakage-free origins of the clean-picker corpus (1,865 of its 4,497 renditions) |
| `picker_safe_origins_2026-08-04.txt` | **SUPERSEDED, do not use** — 319 origins, measured against the invalid training set. The corpus was always fine; the verdict was about its overlap with training, and the repoint changed that. |
| `xl_corpus_subcorpora.tsv` | label→directory map for the XL eval corpus |
| `xl_nasa_leg_dropped_2026-09-07.txt` | the 24 filenames of the XL `nasa` leg, dropped in the repoint — kept so it can be re-acquired deliberately |

`imazen26_eval_files.tsv` was removed in the repoint. `ZENSR_EVAL_PIN` still
overrides the canonical split with a local two-column `dir<TAB>filename` list if
a one-off eval needs it.
