#!/usr/bin/env python3
"""The canonical imazen-26 corpus: github.com/imazen/imazen-26.

**The repository is the corpus.** Not a directory that happens to hold images —
the repo carries the provenance manifests, the canonical train/validate/test
split, the variant-set registry and the generation tooling, and it versions them
as one unit. Image bytes are NOT in git; they are served from public R2 and
synced into the (gitignored) class folders. It moved out of `imazen/codec-corpus`
on 2026-08-23.

**Every other imazen-26 on this machine is stale or invalid** (user directive,
2026-09-07): `/mnt/v/imazen-26*` in any form, and `~/work/codec-corpus/imazen-26`,
which is the pre-move location and two weeks behind. Do not read from them. They
may be used as a byte cache only when every file is verified against this repo's
`sha256` column first — that is how this checkout was populated (2,160/2,160
verified, 0 downloaded).

Nothing here re-derives what the repo already states. The split is
`manifests/split_map.tsv`, the URLs are columns, the hashes are columns. Where
this module adds anything it is because the repo told it to — see
`corpus_split.py` for the one case (near-duplicate grouping).
"""
import csv
import os

# The checkout moved from `~/work/imazen-26` to `~/work/zen/imazen-26` on
# 2026-09-08 17:44, which broke every tool that hardcoded the old path (measured:
# `build_xl_corpus.sh` exited 1 rather than building). So resolve rather than
# hardcode — but only over locations that are the canonical repo. Neither
# `/mnt/v/imazen-26*` nor `~/work/codec-corpus/imazen-26` is a candidate: they are
# invalid and stale respectively, and a "helpful" fallback onto one of them would
# silently train and evaluate on the wrong corpus.
_CANDIDATES = ("~/work/zen/imazen-26", "~/work/imazen-26")


def _resolve_repo():
    env = os.environ.get("IMAZEN26_REPO")
    if env:
        return os.path.expanduser(env)
    for c in _CANDIDATES:
        c = os.path.expanduser(c)
        if os.path.exists(os.path.join(c, "manifests", "split_map.tsv")):
            return c
    return os.path.expanduser(_CANDIDATES[0])


REPO = _resolve_repo()
MANIFEST = os.path.join(REPO, "CORPUS-MANIFEST.tsv")
SPLIT_MAP = os.path.join(REPO, "manifests", "split_map.tsv")

# Kept ONLY to translate the old flat subcorpus names that still appear in
# zensr's history and in ZENSR_SUBS invocations. The canonical corpus does not
# use these names; `content_class` (the folder) is the real key.
OLD_TO_CANONICAL = {
    "lilith": ["1000-lilith-photos-general", "1200-lilith-interiors",
               "1400-lilith-nature", "1600-lilith-food"],
    # NOT the same pictures — half the old `screen` subcorpus was never in the
    # canonical corpus. A content class in common, not a rename.
    "screen": ["8000-lilith-mobile-screenshots", "8100-lilith-web-screenshots"],
    "internet-archive-scans": ["6600-ia-scans-manuscript-illustrations",
                               "6800-ia-scans-manuscript-text"],
    "national-park-service": ["5000-national-park-service-brochures"],
    "unsplash-people": ["2000-unsplash-people"],
    "unsplash-renders": ["2200-unsplash-renders"],
    "unsplash-textures": ["2400-unsplash-textures"],
    # Did not survive curation. Raises rather than returning empty, so a caller
    # cannot silently train on a corpus missing a class it asked for.
    "office-documents": None,
}


def _require_repo():
    if not os.path.exists(SPLIT_MAP):
        raise SystemExit(
            f"canonical imazen-26 not found at {REPO} (no manifests/split_map.tsv).\n"
            f"  git clone https://github.com/imazen/imazen-26 {REPO}\n"
            f"then sync the image bytes per its ACCESS.md. Set IMAZEN26_REPO to "
            f"override the location. Do NOT substitute /mnt/v/imazen-26* or "
            f"~/work/codec-corpus/imazen-26 — both are invalid.")


def split_rows(bucket=None):
    """Rows of the canonical split. `bucket` in {train, validate, test}.

    Columns: id, split, content_class, path, width, height, format,
    bytes_manifest, bytes_actual, sha256, raw_url, png_v3_sdr_url, png_v3_hdr_url.
    """
    _require_repo()
    if bucket:
        p = os.path.join(REPO, "manifests", f"{bucket}.tsv")
    else:
        p = SPLIT_MAP
    with open(p) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def manifest_rows():
    """`CORPUS-MANIFEST.tsv` — the membership oracle, and the only place the
    `descriptor` column lives (the split TSVs do not carry it)."""
    _require_repo()
    with open(MANIFEST) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def all_folders():
    return sorted({r["content_class"] for r in split_rows()})


def abspath(row):
    """Absolute path to a row's image in this checkout."""
    return os.path.join(REPO, row["path"])


def canonical_for(old_sub):
    """Canonical folders for an old flat subcorpus name."""
    if old_sub not in OLD_TO_CANONICAL:
        raise KeyError(f"{old_sub!r} is not an old imazen-26 subcorpus")
    folders = OLD_TO_CANONICAL[old_sub]
    if folders is None:
        raise ValueError(
            f"{old_sub!r} has no canonical equivalent — it did not survive the "
            f"curation pass. Do not substitute a similar-looking folder.")
    return folders


if __name__ == "__main__":
    import collections
    rows = split_rows()
    print(f"canonical imazen-26: {REPO}")
    print(f"{len(rows)} rows\n")
    c = collections.Counter(r["split"] for r in rows)
    for b in ("train", "validate", "test"):
        print(f"  {b:<9} {c[b]:>5}")
    print("\nby content class:")
    per = collections.defaultdict(collections.Counter)
    for r in rows:
        per[r["content_class"]][r["split"]] += 1
    print(f"  {'class':<42} {'train':>6}{'val':>6}{'test':>6}")
    for f in sorted(per):
        k = per[f]
        print(f"  {f:<42} {k['train']:>6}{k['validate']:>6}{k['test']:>6}")
    n_missing = sum(1 for r in rows if not os.path.exists(abspath(r)))
    print(f"\nrows whose image is absent from this checkout: {n_missing}")
