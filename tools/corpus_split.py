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

**The corpus repo's list is necessary but not sufficient** — measured, not
assumed. After applying exactly the groups it enumerates, **105 multi-page
documents still had pages on both sides of the split**, including 12 of the 20
NOAA hurricane advisories (pages of one storm's report, consecutive ids) and 61
of the web-screenshot sites. Those are not in its list. So the base key here is
the exact descriptor within a folder, loosened per folder only where it must be.
Result: **zero groups straddle the split**, no content class loses its validate
or test bucket, and the buckets move from 50/30/19% to 52/30/17% — the test share
is what removing the leakage costs.

The groups, and how each is keyed (verified against the manifest 2026-09-08):

1. `6000-lilith-scans-public-patents` — 3 patents x 3 scan variants (1-bit
   original, colour rescan, grey rescan) of the SAME pages. Key: the patent and
   page, taken from the path's variant directory (`lynn_conway_us5046022_*`) and
   the descriptor's `_pNNN`. 113 files -> 43 groups.
2. `5000-national-park-service-brochures` — `color/` and `grayscale/` renders of
   the same brochures. Key: the descriptor with the `_color` / `_grayscale`
   token removed. 59 files -> 33 groups.
3. `8100-lilith-web-screenshots` — the same URL at up to 6 viewports, AND
   multiple pages per capture. Key: the site (descriptor minus `_dprN_pageN`).
   The repo's list covers only the viewport half; the page half straddled the
   split for 61 sites.
4. `5300-noaa-hurricane-documents` — **not in the repo's list.** Multi-page
   advisories, 3 pages each, consecutive ids; 12 of 20 straddled. Key: the
   advisory (descriptor minus `_pNN`).
5. Everywhere else — the exact descriptor. Catches the plain duplicates: three
   `pink-rose-flower` shots in `1400-lilith-nature`, two `ornate-painted-ceiling`
   in `1200-lilith-interiors`, repeated `9226` product renders.
6. `6600` + `6800` IA scans — illustration plates and text pages drawn from the
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

    The base rule is **the exact descriptor within a folder**, because in this
    corpus an identical descriptor means the same subject: three
    `1400-lilith-nature/pink-rose-flower` files are one flower, and they were
    landing in different buckets. Four folders need a looser key than that, and
    each is loosened only as far as it has to be — see the module docstring.

    Deliberately NOT document level everywhere: with 3 patents and 6 IA works,
    grouping by document collapses those classes to one bucket each (measured:
    patents 104 train / 0 validate / 9 test).
    """
    folder, desc, path = row["folder"], row["descriptor"].strip(), row["path"]
    # An empty descriptor identifies nothing — all 28 `2000-unsplash-people`
    # rows have one, and keying on it merges 28 unrelated photographs.
    if not desc:
        return None
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
        # The viewport is a path directory; the descriptor carries the DPR and
        # the page (`archives-exhibits_dpr1_page2`). Group by the SITE: pages of
        # one capture share chrome, typography and palette, and 61 sites had
        # pages on both sides of the split.
        return (folder, re.sub(r"_dpr\d+(_page\d+)?$", "", desc))
    if folder == "5300-noaa-hurricane-documents":
        # Pages of one advisory: same storm, same scan, consecutive ids. 12 of
        # the 20 advisories straddled the split before this. Grouping by
        # document leaves 20 groups -> 22/15/7, which is healthy.
        return (folder, re.sub(r"_p\d+$", "", desc))
    if ia_group and folder in IA_FOLDERS:
        return ("ia-scans", desc.split("-")[0])
    return (folder, desc)


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


def write_split(path, dedup=True, ia_group=False):
    """Emit the effective split as `path<TAB>bucket`.

    This file is the contract between the Python training side and the Rust eval
    side. Both MUST read the same one: the near-duplicate same-bucketing moves
    309 files, and 180 of them cross between held-out and train — if Rust read
    the corpus repo's raw buckets while Python trained on these, it would score
    180 files the model had been trained on. Measured, not hypothesised.

    Generated, not committed: it is a deterministic function of the corpus repo
    (`just split` regenerates it). Consumers fail loudly when it is missing
    rather than falling back to the raw canonical buckets, because that fallback
    IS the 180-file disagreement.
    """
    sm = split_map(dedup=dedup, ia_group=ia_group)
    with open(path, "w") as f:
        f.write("# Effective imazen-26 split for zensr. GENERATED — do not edit,\n"
                "# do not commit. Regenerate: just split\n"
                "#\n"
                "# Canonical buckets from github.com/imazen/imazen-26\n"
                "# (manifests/split_map.tsv), with near-duplicate groups\n"
                "# same-bucketed per tools/corpus_split.py. Both the Python\n"
                "# trainer and the Rust eval harness read THIS file.\n"
                "# path\tbucket\n")
        for k in sorted(sm):
            f.write(f"{k}\t{sm[k]}\n")
    return len(sm)


if __name__ == "__main__":
    if "--write" in sys.argv:
        out = sys.argv[sys.argv.index("--write") + 1]
        n = write_split(out)
        c = collections.Counter(split_map().values())
        print(f"wrote {out}: {n} rows "
              f"(train {c['train']}, validate {c['validate']}, test {c['test']})")
        raise SystemExit(0)
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
