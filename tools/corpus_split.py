#!/usr/bin/env python3
"""The canonical imazen-26 split, plus the near-duplicate handling zensr needs.

**The split is not computed here.** It is `manifests/split_map.tsv` in the
canonical corpus repo (github.com/imazen/imazen-26), whose rule is the last digit
of the 4-digit image id — {0,2,4,6,8} train, {1,3,5} validate, {7,9} test — and
which is identical to zenmetrics `scripts/picker/origin_split.py`. Every
derivative of an image inherits the image's bucket, and datasets join on `id`.
Canonical totals: 1,084 train / 658 validate / 418 test.

zensr previously re-derived a split of its own (twice: "first 8 sorted ∪ a pinned
list of 64", and then an origin-grouping of my own invention). Both are gone. The
repo states the split; this module reads it.

**The one addition, and it is the repo's own instruction.** `manifests/README.md`
says: *"The split is by id, so related images with different ids can land in
different buckets. If your task is sensitive to near-duplicate leakage, drop or
same-bucket these enumerable groups"* — and then enumerates exactly four. zensr
restores compression damage, so scanner texture, paper stock and page furniture
are precisely the confound that makes a near-duplicate leak matter: a model that
has seen page 2 of a document has effectively seen page 3. So zensr same-buckets
them, by the group's minimum id, which keeps the canonical rule as the decider.

The four groups, and how each is keyed (verified against the manifest 2026-09-08):

1. `6000-lilith-scans-public-patents` — 3 patents x 3 scan variants (1-bit
   original, colour rescan, grey rescan) of the SAME pages. Key: the patent and
   page, taken from the path's variant directory (`lynn_conway_us5046022_*`) and
   the descriptor's `_pNNN`. 113 files -> 43 groups.
2. `5000-national-park-service-brochures` — `color/` and `grayscale/` renders of
   the same brochures. Key: the descriptor with the `_color` / `_grayscale`
   token removed. 59 files -> 33 groups.
3. `8100-lilith-web-screenshots` — the same URL captured at up to 6 viewports
   (the viewport is a path directory, not part of the descriptor). Key: the
   descriptor alone. 370 files -> 81 groups.
4. `6600` + `6800` IA scans — illustration plates and text pages drawn from the
   same six source works. Key: the source work. **This one is OFF by default**,
   and that is a measured decision, not an oversight: there are only six works,
   so grouping by work leaves both IA folders with **no test bucket at all**
   (6600 -> 31 train / 5 validate / 0 test; 6800 -> 35 / 1 / 0). With 72 files
   there is no arrangement that has both zero leakage and a populated held-out
   set. Between an IA number that is slightly optimistic and no IA number at
   all, the optimistic one is more useful — provided it is labelled, which is
   what this comment and `docs/CORPUS-REPOINT-IMPACT.md` are for. Pass
   `ia_group=True` to take the other trade.

   The leak it admits is mild in kind: an engraving plate and a page of text from
   the same book share a scanner and a paper stock, not content.

Everything outside those folders is left exactly as the canonical split has it.
Use `split_map(dedup=False)` for the unmodified canonical buckets.

**A correction to the corpus repo's own description.** `manifests/README.md`
describes group 2 as "`color/` + `grayscale/` renders of the same brochures".
Measured against `CORPUS-MANIFEST.tsv` on 2026-09-08: the two directories hold
**59 distinct brochures** (33 colour, 26 greyscale) with **zero** shared stems
and 59 distinct `original_filename` values — they are different brochures, not
two renders of one set. The real near-duplicate structure there is *pages of one
brochure*, which is what the key below uses. Reported upstream.
"""
import collections
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from imazen26_canonical import manifest_rows, split_rows  # noqa: E402

IA_FOLDERS = ("6600-ia-scans-manuscript-illustrations",
              "6800-ia-scans-manuscript-text")


def group_key(row, ia_group=False):
    """The near-duplicate group a manifest row belongs to, or None.

    Groups duplicate RENDERS of one page/plate — the level the corpus repo
    enumerates. Deliberately NOT document level: with 3 patents and 6 IA works,
    grouping by document collapses those classes to one bucket each (measured:
    patents 104 train / 0 validate / 9 test).
    """
    folder, desc, path = row["folder"], row["descriptor"].strip(), row["path"]
    if folder == "6000-lilith-scans-public-patents":
        parts = path.split("/")
        variant_dir = parts[1] if len(parts) > 2 else ""
        # lynn_conway_us5046022_1bitoriginal -> lynn_conway_us5046022, so the
        # 1-bit / colour-rescan / grey-rescan copies of one page group together.
        patent = re.sub(r"_(1bitoriginal|printrescan\w+)$", "", variant_dir)
        page = re.search(r"_p(\d+)", desc)
        return (folder, patent, page.group(1) if page else desc)
    if folder == "5000-national-park-service-brochures":
        # Strip the colour token (`_color` / `_gray` — note the descriptor says
        # `gray` while the directory says `grayscale`) and the page index, so the
        # group is the brochure. No colour pair actually exists today; keying it
        # correctly costs nothing and guards the next import.
        return (folder, re.sub(r"_(color|gray|grayscale)(?=_|$)", "",
                               re.sub(r"_p\d+$", "", desc)))
    if folder == "8100-lilith-web-screenshots":
        # The viewport is a path directory, not part of the descriptor, so the
        # descriptor alone groups one capture across up to 6 viewports.
        return (folder, desc)
    if ia_group and folder in IA_FOLDERS:
        return ("ia-scans", desc.split("-")[0])
    return None


def split_map(dedup=True, ia_group=False):
    """`{path: 'train'|'validate'|'test'}`.

    dedup=True (default) same-buckets the enumerated near-duplicate groups to the
    bucket of the group's minimum id, keeping the canonical rule as the decider.
    dedup=False returns the canonical buckets untouched.
    ia_group=True additionally groups the IA scans by source work, at the cost of
    both IA folders' test buckets — see the module docstring.
    """
    canon = {r["path"]: r["split"] for r in split_rows()}
    if not dedup:
        return canon
    rows = manifest_rows()
    groups = collections.defaultdict(list)
    for r in rows:
        k = group_key(r, ia_group=ia_group)
        if k is not None:
            groups[k].append(r)
    out = dict(canon)
    for members in groups.values():
        lead = min(members, key=lambda m: int(m["number"]))
        bucket = canon[lead["path"]]
        for m in members:
            out[m["path"]] = bucket
    return out


if __name__ == "__main__":
    canon = split_map(dedup=False)
    ded = split_map(dedup=True)
    rows = manifest_rows()
    groups = collections.defaultdict(list)
    for r in rows:
        k = group_key(r)
        if k is not None:
            groups[k].append(r)
    per_folder = collections.Counter()
    for k in groups:
        per_folder[k[0]] += 1
    n_in = sum(len(v) for v in groups.values())
    print(f"near-duplicate groups: {n_in} files -> {len(groups)} groups")
    for f in sorted(per_folder):
        n = sum(len(v) for k, v in groups.items() if k[0] == f)
        print(f"  {f:<42} {n:>4} files -> {per_folder[f]:>3} groups")
    moved = [p for p in canon if canon[p] != ded[p]]
    print(f"\nfiles moved by same-bucketing: {len(moved)}")
    for label, m in (("canonical", canon), ("deduped", ded)):
        c = collections.Counter(m.values())
        tot = sum(c.values())
        print(f"  {label:<10} train {c['train']:>5} ({100*c['train']/tot:.0f}%)  "
              f"validate {c['validate']:>4} ({100*c['validate']/tot:.0f}%)  "
              f"test {c['test']:>4} ({100*c['test']/tot:.0f}%)")
