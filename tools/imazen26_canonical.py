#!/usr/bin/env python3
"""The canonical imazen-26 corpus, and how the old flat layout maps onto it.

**Why this file exists.** zensr trained and evaluated against `/mnt/v/imazen-26`,
the *pre-curation acquisition* corpus, while every other consumer migrated to the
curated `~/work/codec-corpus/imazen-26`. User ruling 2026-08-05: the curated one
is the only valid imazen-26. Background: `docs/CORPUS-REPOINT-HANDOFF.md`,
evidence in `benchmarks/imazen26_contamination_audit_2026-08-05.md`.

The mapping lives here, once, because it is **not 1:1 and not inferable**. The
old flat `lilith` spans four canonical folders; the old `screen` spans two that
are not the same images; two old subcorpora have no canonical home at all. A
caller that guesses will guess differently from the next caller.

Everything below is derived from `CORPUS-MANIFEST.tsv` — 2,160 rows, 21 folders,
every row's file verified present on disk (2026-09-07). The manifest's `source`
column is the acquisition provenance and is what the mapping keys on; do not
re-derive it from filenames, which were changed by the curation pass.

`nope/` (406 images) is the licence-excluded split-off. It carries **zero**
manifest rows and must never be walked.
"""
import csv
import os

CANONICAL_ROOT = os.path.expanduser("~/work/codec-corpus/imazen-26")
MANIFEST = os.path.join(CANONICAL_ROOT, "CORPUS-MANIFEST.tsv")
# Never walk this: licence-excluded material, deliberately outside the manifest.
EXCLUDED_DIRS = {"nope", "scripts"}

# Old flat subcorpus -> canonical folders. Counts are canonical-side, measured
# 2026-09-07. `None` means the material did not survive curation.
OLD_TO_CANONICAL = {
    # 319 canonical images against 8 old pinned-eval picks, only 2 of which
    # match by name — the curation renamed most of this subcorpus.
    "lilith": ["1000-lilith-photos-general", "1200-lilith-interiors",
               "1400-lilith-nature", "1600-lilith-food"],
    # 402 canonical images, but NOT the same pictures: dev's audit found half of
    # the old `screen` subcorpus absent from the canonical corpus entirely
    # (219 of the 250 non-carried files). Treat as a different subcorpus that
    # happens to share a content class, not as a renamed one.
    "screen": ["8000-lilith-mobile-screenshots", "8100-lilith-web-screenshots"],
    "internet-archive-scans": ["6600-ia-scans-manuscript-illustrations",
                               "6800-ia-scans-manuscript-text"],
    "national-park-service": ["5000-national-park-service-brochures"],
    "unsplash-people": ["2000-unsplash-people"],
    "unsplash-renders": ["2200-unsplash-renders"],
    "unsplash-textures": ["2400-unsplash-textures"],
    # NO CANONICAL HOME. 31 acquisition files (US Federal Register + IRS forms);
    # zero manifest rows carry them, and 0 of its 8 pinned eval files resolve.
    # The nearest canonical material (5200-epa, 5300-noaa, 6000-uspto) is
    # different documents from different sources — substituting it would be a
    # silent corpus change, not a repoint.
    "office-documents": None,
}

# Canonical folders with no counterpart in the old corpus — content zensr has
# never trained or evaluated on. 1,257 images, more than the old corpus held in
# total (1,068), and several are whole content classes the router's classifier
# has never seen.
NEW_IN_CANONICAL = {
    "9226-lilith-ai-products": 749,
    "7000-lilith-plots": 126,
    "6000-lilith-scans-public-patents": 113,
    "9000-lilith-ai-clipart": 86,
    "9094-lilith-ai-illustrations": 75,
    "5300-noaa-hurricane-documents": 44,
    "5200-epa-climate-impact-2021-report": 25,
    "3300-met-museum-photos": 24,
    "3000-art-institute-of-chicago-photos": 15,
}

# The default training set: every canonical folder except the excluded dirs.
# Deliberately NOT the old eight remapped — the curation added content classes
# worth training on, and restricting to the old set would preserve a limitation
# that only ever existed because of the wrong root.
def all_folders():
    return sorted({r["folder"] for r in manifest_rows()})


def manifest_rows():
    """Every canonical image, as manifest dicts. The manifest is authoritative:
    all 2,160 rows were verified present on disk 2026-09-07."""
    with open(MANIFEST) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def files_in(folders=None):
    """Absolute paths for the given canonical folders (default: all)."""
    want = set(folders) if folders else None
    out = []
    for r in manifest_rows():
        if r["folder"] in EXCLUDED_DIRS:
            continue
        if want is None or r["folder"] in want:
            out.append(os.path.join(CANONICAL_ROOT, r["path"]))
    return sorted(out)


def canonical_for(old_sub):
    """Canonical folders for an old flat subcorpus name.

    Raises rather than returning empty for a subcorpus that did not survive:
    a caller that silently gets zero files would train on a corpus quietly
    missing a content class.
    """
    if old_sub not in OLD_TO_CANONICAL:
        raise KeyError(f"{old_sub!r} is not an old imazen-26 subcorpus")
    folders = OLD_TO_CANONICAL[old_sub]
    if folders is None:
        raise ValueError(
            f"{old_sub!r} has no canonical equivalent — it did not survive the "
            f"curation pass. See OLD_TO_CANONICAL in this file. Do not "
            f"substitute a similar-looking folder."
        )
    return folders


if __name__ == "__main__":
    rows = manifest_rows()
    folders = {}
    for r in rows:
        folders[r["folder"]] = folders.get(r["folder"], 0) + 1
    print(f"canonical root: {CANONICAL_ROOT}")
    print(f"{len(rows)} manifest rows, {len(folders)} folders\n")
    for k in sorted(folders, key=lambda k: -folders[k]):
        tag = "  NEW (never trained on)" if k in NEW_IN_CANONICAL else ""
        print(f"  {folders[k]:>5}  {k}{tag}")
    print("\nold flat subcorpus -> canonical:")
    for old, new in OLD_TO_CANONICAL.items():
        if new is None:
            print(f"  {old:<24} -- NO CANONICAL HOME")
        else:
            n = sum(folders.get(f, 0) for f in new)
            print(f"  {old:<24} -> {n:>4} images in {', '.join(new)}")
