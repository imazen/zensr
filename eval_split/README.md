# eval_split — where the split actually lives

**Not here.** The canonical imazen-26 train/validate/test split is part of the
corpus repository, `github.com/imazen/imazen-26`, at
`manifests/{split_map,train,validate,test}.tsv`. The rule is the last digit of
the 4-digit image id — {0,2,4,6,8} train, {1,3,5} validate, {7,9} test —
identical to zenmetrics `scripts/picker/origin_split.py`. Totals: 1,084 / 658 /
418 over 2,160 images.

zensr reads it directly: `zensr_bench::canonical_holdout()` in Rust,
`tools/corpus_split.py` in Python. Point `IMAZEN26_REPO` at the checkout if it is
not at `~/work/imazen-26`.

**Why nothing is vendored here.** zensr kept its own eval list twice — a
hand-maintained 64-file pin (`imazen26_eval_files.tsv`, keyed to the deleted
`/mnt/v/imazen-26` layout), and then a re-derived split of its own. Both drifted
from the corpus, and the first leaked training images into an eval twice. A copy
of someone else's split is a copy that will be wrong later.

`tools/corpus_split.py` adds exactly one thing on top of the canonical buckets:
it same-buckets the near-duplicate groups the corpus repo's own
`manifests/README.md` tells near-duplicate-sensitive consumers to handle
(duplicate renders of one patent page, one brochure, one web capture across
viewports). 210 files move; the bucket proportions do not.

## What IS here

| file | what |
|---|---|
| `picker_safe_origins_2026-08-04.txt` | 319 leakage-free origins of the clean-picker corpus, which never touched the invalid root |
| `xl_corpus_subcorpora.tsv` | label→directory map for the XL eval corpus |
| `xl_nasa_leg_dropped_2026-09-07.txt` | the 24 filenames of the XL `nasa` leg, dropped in the repoint — kept so it can be re-acquired deliberately |

`imazen26_eval_files.tsv` was removed in the repoint. `ZENSR_EVAL_PIN` still
overrides the canonical split with a local two-column `dir<TAB>filename` list if
a one-off eval needs it.
