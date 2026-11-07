# Split authorities and manifest fields

## Published imazen-26 ID split

The corpus-owned split used here is **imazen/imazen-26**, revision
`187fbf338ce08e8e6654db7f04ddae58d5263da2`, file
[`manifests/split_map.tsv`](https://github.com/imazen/imazen-26/blob/187fbf338ce08e8e6654db7f04ddae58d5263da2/manifests/split_map.tsv).
This names a frozen revision, not whichever corpus revision happens to be newest.

The last digit of the source's four-digit ID decides membership: 0/2/4/6/8
are `train`, 1/3/5 are `validate`, and 7/9 are `test`. Counts are
**1,084 / 658 / 418 source images**. Derivatives inherit their source ID's bucket.
See the corpus's
[published instructions](https://github.com/imazen/imazen-26/blob/187fbf338ce08e8e6654db7f04ddae58d5263da2/manifests/README.md).

## zensr historical reassignment

The historical zensr generator instead used a downstream policy in
[`tools/corpus_split.py`](https://github.com/imazen/zensr/blob/4a33019/tools/corpus_split.py).
It grouped related images and reassigned members to the lowest-ID member's
bucket. Its counts are **1,151 / 648 / 361**. This is **zensr's policy**, not
the published imazen-26 ID split. It assigned 103 imazen-26 validation and 77
imazen-26 test sources to its eligible training pool. The original canon4
model inherited that policy; the model name does not certify split compliance.

## Pair dataset's internal partition

The original pair arrays reserve 512 of 48,000 rows for internal validation;
47,488 rows are eligible for optimizer training. This is a zensr dataset
partition, not imazen-26's `validate` bucket. Both partitions were selected
from zensr's historical reassigned source pool.

The new preview filters source IDs using the **published imazen-26 ID split**:
39,895 rows remain, comprising 39,503 original optimizer-training rows and
392 original internal-validation rows. All those source IDs belong to
imazen-26 `train`. The separate 8,105-row archive contains sources belonging
to imazen-26 `validate` or `test`. No image ID is reassigned by this filter.

## Existing names and schema

| Existing field or name | Explicit meaning |
|---|---|
| `canonical_split` | Published imazen-26 ID bucket at the frozen revision above |
| `effective_split` | zensr historical reassigned bucket from its grouping helper |
| `partition` | Pair-array role: `model_train` or `internal_validation` |
| `canonical-train` CLI policy | Select only source IDs in published imazen-26 `train` |
| `canon4-canonical-train-*.tar.zst` | Preview shards whose source IDs belong to published imazen-26 `train` |
| `canon4-canonical-heldout.tar.zst` | Preview rows whose source IDs belong to published imazen-26 `validate`/`test` |

Existing field names, filenames and bytes remain for compatibility. The word
"canonical" in those names refers only to the **published imazen-26 ID split**.
It does not refer to zensr's historical reassignment or establish source-byte
agreement with older manifests, visual privacy review, or content-family
independence across source IDs.

Routing-study `router_development` and `locked_test` roles are another zensr-owned
selection layer, with origin exclusions and previous-inspection handling. Their
authority is `tools/publication_routing_manifest.py`, not imazen-26's split map.
