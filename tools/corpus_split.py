#!/usr/bin/env python3
"""Origin-level train/val/test split for the canonical imazen-26 corpus.

Named `corpus_split`, not `origin_split`, on purpose: zenmetrics owns a module
with the latter name and this one imports it. Sharing the name made this file
shadow the very rule it was loading.

The digit rule is NOT defined here. It lives in zenmetrics
(`scripts/picker/origin_split.py`), whose header says *"Import this everywhere
... do not re-implement the rule"* — and zensr has already paid for ignoring
that once. This module imports it and adds exactly one thing on top: **grouping**.

**Why grouping is needed.** The canonical rule buckets by the last digit of a
file's leading numeric id, which assumes one id per picture. In the canonical
corpus that assumption breaks: three of the 44 noaa files are three pages of one
hurricane report, carrying three consecutive ids. Pages of one document share a
scanner, a typeface and a paper stock, so splitting per-file puts near-duplicates
on both sides and "held out" stops meaning anything. So: group first, then apply
the canonical rule to the group's **minimum** id, and every member inherits it.

The grouping key is `(folder, descriptor)` straight out of `CORPUS-MANIFEST.tsv`.

**The empty-descriptor trap.** All 28 rows of `2000-unsplash-people` have an
empty `descriptor` — theirs lives in the filename (`by-alexander-aguero-...`).
Keying naively on `(folder, "")` merges 28 unrelated photographs into one origin
and dumps them in a single bucket. Fall back to the filename stem when the
descriptor is empty. This is the same shape as the bug that once merged 37
distinct CID22 images by stripping a trailing `-\\d+` from `pexels-photo-1029599`.

Measured on the canonical manifest (2026-09-07): 2,160 files → 1,884 origins,
104 multi-file, 380 files (18%) in a multi-file group, largest group 7.
"""
import collections
import importlib.util
import os
import sys

# Load the canonical rule BY PATH, under a distinct module name. A plain
# `from origin_split import split_of` resolves to whichever `origin_split` is
# first on sys.path — which, when this file was itself named `origin_split.py`,
# was this file importing itself. Hence both the rename and the explicit path.
_CANON = os.path.expanduser(os.environ.get(
    "ZENMETRICS_ORIGIN_SPLIT",
    "~/work/zen/zenmetrics/scripts/picker/origin_split.py"))
if not os.path.exists(_CANON):  # pragma: no cover
    raise SystemExit(
        f"canonical split rule not found at {_CANON}. Set "
        "ZENMETRICS_ORIGIN_SPLIT to zenmetrics' scripts/picker/origin_split.py. "
        "Do NOT re-implement the rule locally — that is exactly the mistake "
        "this import exists to prevent.")
_spec = importlib.util.spec_from_file_location("zenmetrics_origin_split", _CANON)
_canon = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_canon)
split_of = _canon.split_of  # the CANONICAL rule, unmodified

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from imazen26_canonical import manifest_rows  # noqa: E402


def group_key(row):
    """The origin a manifest row belongs to. Descriptor if it has one, else the
    filename stem (see the 2000-unsplash-people trap in the module docstring)."""
    desc = row["descriptor"].strip()
    if not desc:
        desc = os.path.basename(row["path"]).rsplit(".", 1)[0]
    return (row["folder"], desc)


def origins(rows=None):
    """`{(folder, descriptor): [row, ...]}` for the canonical corpus."""
    g = collections.defaultdict(list)
    for r in rows if rows is not None else manifest_rows():
        g[group_key(r)].append(r)
    return g


def split_map(rows=None):
    """`{manifest path: 'train'|'val'|'test'}`, origin-level.

    The bucket comes from the canonical rule applied to the group's MINIMUM id,
    and every member of the group inherits it — that inheritance is the whole
    point, so no derivative of an origin can cross the split.
    """
    out = {}
    for _, members in origins(rows).items():
        lead = min(int(m["number"]) for m in members)
        bucket = split_of(str(lead))
        if bucket is None:
            raise ValueError(f"unsplittable origin id {lead!r} — the canonical "
                             f"rule returned no bucket")
        for m in members:
            out[m["path"]] = bucket
    return out


if __name__ == "__main__":
    rows = manifest_rows()
    g = origins(rows)
    sizes = collections.Counter(len(v) for v in g.values())
    multi = sum(n for s, n in sizes.items() if s > 1)
    in_multi = sum(s * n for s, n in sizes.items() if s > 1)
    print(f"{len(rows)} files -> {len(g)} origins")
    print(f"  {multi} multi-file origins, {in_multi} files "
          f"({100 * in_multi / len(rows):.0f}%) in one, largest {max(sizes)}")
    sm = split_map(rows)
    c = collections.Counter(sm.values())
    print("\nbucket   files  share  canonical")
    for b, target in (("train", 50), ("val", 30), ("test", 20)):
        print(f"  {b:<6} {c[b]:>5}  {100 * c[b] / len(sm):>4.0f}%  {target}%")
    # Per-folder, so a repoint can see which content classes land where.
    print("\nper-folder train/val/test:")
    per = collections.defaultdict(collections.Counter)
    for r in rows:
        per[r["folder"]][sm[r["path"]]] += 1
    for f in sorted(per):
        k = per[f]
        print(f"  {f:<42} {k['train']:>4} {k['val']:>4} {k['test']:>4}")
