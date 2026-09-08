#!/usr/bin/env python3
"""Which clean-picker-corpus origins are free of dejpeg training content?

I first concluded the canonical picker datasets were unusable for dejpeg eval —
built from imazen-26, which dejpeg trains on, with "no origin→path map available
to filter by". **That was wrong on the second half.** The map exists:
`clean-picker-corpus-2026-06-26/_provenance.tsv` carries `source_sha256` for
every rendition, and following it resolves all 414 origins to a source file.

Following it through also reverses the verdict: **77% of the corpus is
leakage-free**, not unusable.

The chain, and why each step is needed:

1. `_provenance.tsv` → `source_sha256` per rendition.
2. sha → source file, hashing both `/mnt/v/imazen-26` and
   `/mnt/v/output/imazen-26-png`. Only 96 of 414 resolve against imazen-26; the
   other 318 live in imazen-26-png, which is a **larger, differently organised**
   collection (2,639 files in numbered dirs like `9226-lilith-ai-products`
   versus 1,068 in flat dirs like `lilith`).
3. Classify against the dejpeg training set — the 8 trained subcorpora minus
   (pinned ∪ first-8-sorted).

Step 3 needs both an exact and a perceptual test, and getting this wrong is easy
in both directions:

- Comparing **directory names** across the two roots is wrong. It looks like
  `2000-unsplash-people` is "not a trained subcorpus" when it is the same
  content as `unsplash-people`, which is trained on. That mistake reports 325
  origins safe.
- Comparing **filename stems** resolves only part of it: stem preservation
  between the two roots is 42%, and six subcorpora preserve none at all
  (`benchmarks/imazen26_naming_break_2026-08-05.md`). An earlier version of this
  docstring said stems matched ZERO — that was a bug in my comparison, not the
  data: these files carry a `.sdr.png` double extension and I split on the last
  dot only.
- Comparing **bytes** resolves 59%. An earlier version said none were
  byte-identical; I never measured that and inferred it from "imazen-26-png is
  re-encoded".

So the only thing that works for those 318 is a content fingerprint. A 16×16
luma thumbnail with mean |Δ| < 3/255 catches re-encodes and format conversions
of the same scene, which is what leakage means here — the model saw the picture,
not the file.

Result (2026-08-04): 319 safe, 81 exact training files, 14 near-duplicates
→ 3,452 of 4,497 renditions usable. Those renditions are **size-diverse**
(`scale36x64` upward), which the XL corpus is not — every XL image is a 512
crop, and the sweep discipline asks for 16-20 log-spaced sizes for anything a
model is fitted on.

Usage: picker_leakage_audit.py [--write eval_split/picker_safe_origins_<date>.txt]
"""
import collections
import hashlib
import os
import sys
import warnings

warnings.filterwarnings("ignore")
from PIL import Image  # noqa: E402

Image.MAX_IMAGE_PIXELS = None

PICKER = "/mnt/v/output/clean-picker-corpus-2026-06-26"
ROOTS = [CANONICAL_ROOT, "/mnt/v/output/imazen-26-png-v3"]
# Repointed 2026-09-07: the training set is now the TRAIN BUCKET of the canonical
# corpus, not eight flat subcorpora of the deleted /mnt/v/imazen-26 root. Both
# halves changed — the images and the exclusion rule — so a verdict from before
# this date answers a question about a corpus that no longer exists.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from corpus_split import split_map  # noqa: E402
from imazen26_canonical import CANONICAL_ROOT  # noqa: E402
# Mean per-pixel luma difference on a 16x16 thumbnail, below which two images are
# the same scene. Loose enough for a re-encode or a format conversion, tight
# enough that distinct photos do not collide.
NEAR = 3.0


def thumb(p):
    try:
        return tuple(Image.open(p).convert("L").resize((16, 16)).getdata())
    except Exception:
        return None


def dist(a, b):
    return sum(abs(x - y) for x, y in zip(a, b)) / 256.0


def image_files(root):
    for base, _, names in os.walk(root):
        for n in sorted(names):
            if n.lower().endswith((".png", ".jpg", ".jpeg")):
                yield os.path.join(base, n)


def training_set():
    """Canonical-corpus files in the TRAIN bucket — see leakage_audit.py."""
    return {os.path.join(CANONICAL_ROOT, p)
            for p, b in split_map().items() if b == "train"}


def main():
    train = training_set()
    train_fp = [(t, p) for t, p in ((thumb(p), p) for p in train) if t]
    print(f"dejpeg training: {len(train)} files, {len(train_fp)} fingerprinted")

    sha2origin = {}
    rows = []
    with open(f"{PICKER}/_provenance.tsv") as f:
        next(f)
        for line in f:
            c = line.rstrip("\n").split("\t")
            rows.append((c[0], c[1]))
            sha2origin[c[2]] = c[1]

    h2p = {}
    for r in ROOTS:
        for p in image_files(r):
            try:
                h = hashlib.sha256(open(p, "rb").read()).hexdigest()
            except Exception:
                continue
            h2p.setdefault(h, p)

    counts = collections.Counter()
    safe = set()
    for sha, origin in sha2origin.items():
        p = h2p.get(sha)
        if p is None:
            counts["unresolved (treated as unsafe)"] += 1
            continue
        if p in train:
            counts["exact training file"] += 1
            continue
        t = thumb(p)
        if t is None:
            counts["undecodable (treated as unsafe)"] += 1
            continue
        if min(dist(t, r) for r, _ in train_fp) < NEAR:
            counts["near-duplicate of training"] += 1
        else:
            counts["dejpeg-safe"] += 1
            safe.add(origin)

    total = sum(counts.values())
    for k, v in counts.most_common():
        print(f"  {k:<34} {v:>4} ({100 * v / total:.0f}%)")
    usable = sum(1 for _, src in rows if src in safe)
    print(f"\nsafe renditions: {usable} of {len(rows)}")

    if "--write" in sys.argv:
        out = sys.argv[sys.argv.index("--write") + 1]
        with open(out, "w") as f:
            f.write("\n".join(sorted(safe)) + "\n")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
