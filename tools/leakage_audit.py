#!/usr/bin/env python3
"""Is a candidate corpus free of dejpeg training content?

Generalises `picker_leakage_audit.py`, which answered this for one corpus. Any
dataset proposed for dejpeg *evaluation* has to clear the same bar, and the bar
is not "different filenames" — it is "the model has not seen this picture".

Three tests, because each catches what the previous misses:

1. **Exact path.** Catches a candidate that literally is a training file.
2. **Content hash.** Catches a byte-identical copy under another name. Neither
   of these is sufficient alone: `imazen-26-png` is a partly re-encoded,
   re-organised copy of imazen-26 that preserves only **42%** of stems and
   **59%** of bytes — six subcorpora preserve none at all — so both tests miss
   a large share of it (`benchmarks/imazen26_naming_break_2026-08-05.md`).
3. **Perceptual fingerprint.** A 16x16 luma thumbnail with mean |Δ| below
   `--near` (default 3/255). This is the one that actually works for re-encodes,
   format conversions and resizes of the same scene.

Resizes are the reason a plain fingerprint is not enough either: a 512px and a
1024px render of one photo are the same picture at different scales, and the
thumbnail normalises scale away, which is exactly what is wanted here.

Reports per-origin as well as per-file, since a split has to be origin-level —
one page of a document leaking implicates every other page of it.

**The fingerprint over-flags document scans — confirm a hit before acting on it.**
Measured 2026-09-07 against the patent corpus: 29 pages flagged across six patent
documents, but only THREE of those documents are actually in the training corpus.
The three true ones matched at distance **0.00** and their filenames agreed
(`US5046022-002.png` against `6002_scans-patents_lynn-conway-us5046022-...`); the
three false ones matched at 2.4-2.7 against pages of an *unrelated* patent, and
two of them had fingerprint stdev around 6.5 — near-blank text pages, which at
16x16 all look alike. 42 of 1,034 training fingerprints are near-blank like that,
so the collision surface is real. Acting on the fingerprint alone would have
discarded 87 clean pages.

So: the fingerprint is a **candidate generator**, not a verdict. Where the corpus
carries identity (a patent number, a manifest id, a source hash), confirm against
it. `--near` trades the two error directions and neither is free; a low threshold
misses re-encodes, a high one eats blank pages.

Usage:
  leakage_audit.py --files <list.txt> [--near 3.0] [--write-safe out.txt]
  leakage_audit.py --glob '/path/**/*.png'
"""
import argparse
import collections
import hashlib
import os
import re
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

Image.MAX_IMAGE_PIXELS = None

# Repointed 2026-09-07: the training set is now the TRAIN BUCKET of the canonical
# corpus, not eight flat subcorpora of the deleted /mnt/v/imazen-26 root. Both
# halves changed — the images and the exclusion rule — so a verdict from before
# this date answers a question about a corpus that no longer exists.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from corpus_split import split_map  # noqa: E402
from imazen26_canonical import REPO  # noqa: E402


def thumb(p):
    try:
        im = Image.open(p).convert("L").resize((16, 16))
        return np.asarray(im, dtype=np.float32).ravel()
    except Exception:
        return None


def sha(p):
    try:
        with open(p, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except Exception:
        return None


def image_files(root):
    for base, _, names in os.walk(root):
        for n in sorted(names):
            if n.lower().endswith((".png", ".jpg", ".jpeg")):
                yield os.path.join(base, n)


def training_files():
    """Canonical-corpus files in the TRAIN bucket — what the model actually sees.

    The old definition was "everything except the first 8 sorted and a pinned
    list", which was a hand-maintained approximation of a split. The canonical
    corpus has ids, so this is now the split itself.
    """
    return {os.path.join(REPO, p)
            for p, b in split_map().items() if b == "train"}


def origin_of(path):
    """Group derivatives of one source. Handles the conventions seen so far:
    a leading content hash with a size suffix (`<hash>_1024sq`), page/figure
    indices, and `WxH` rendition suffixes."""
    stem = os.path.basename(path).rsplit(".", 1)[0]
    m = re.match(r"^([0-9a-f]{8,})_", stem)
    if m:
        return m.group(1)
    stem = re.sub(r"_(p|page)\d+$", "", stem)
    return re.sub(r"[_.]?\d+x\d+(sq)?$|_\d+sq$", "", stem)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files")
    ap.add_argument("--glob")
    ap.add_argument("--near", type=float, default=3.0)
    ap.add_argument("--write-safe")
    a = ap.parse_args()

    if a.files:
        cands = [l.strip() for l in open(a.files) if l.strip()]
    elif a.glob:
        import glob
        cands = glob.glob(a.glob, recursive=True)
    else:
        sys.exit("need --files or --glob")

    train = training_files()
    tfp, tsha = [], set()
    for p in train:
        t = thumb(p)
        if t is not None:
            tfp.append(t)
        s = sha(p)
        if s:
            tsha.add(s)
    T = np.stack(tfp) if tfp else np.zeros((0, 256), np.float32)
    print(f"dejpeg training: {len(train)} files, {len(T)} fingerprinted")
    print(f"candidates: {len(cands)}\n")

    counts = collections.Counter()
    safe, by_origin = [], collections.defaultdict(list)
    for p in cands:
        o = origin_of(p)
        if p in train:
            counts["exact training path"] += 1
            by_origin[o].append(False)
            continue
        if sha(p) in tsha:
            counts["byte-identical to training"] += 1
            by_origin[o].append(False)
            continue
        t = thumb(p)
        if t is None:
            counts["undecodable (unsafe)"] += 1
            by_origin[o].append(False)
            continue
        # Mean |Δ| against every training fingerprint at once.
        if len(T) and float(np.abs(T - t).mean(axis=1).min()) < a.near:
            counts["near-duplicate of training"] += 1
            by_origin[o].append(False)
        else:
            counts["SAFE"] += 1
            safe.append(p)
            by_origin[o].append(True)

    tot = sum(counts.values())
    for k, v in counts.most_common():
        print(f"  {k:<30} {v:>6} ({100 * v / tot:.1f}%)")

    # An origin is safe only if every one of its derivatives is.
    clean = [o for o, v in by_origin.items() if all(v)]
    print(f"\norigins: {len(by_origin)} total, {len(clean)} fully clean")
    print(f"files:   {tot} total, {counts['SAFE']} safe")

    if a.write_safe:
        with open(a.write_safe, "w") as f:
            f.write("\n".join(sorted(p for p in safe if origin_of(p) in clean)) + "\n")
        print(f"wrote {a.write_safe} (safe files from fully-clean origins only)")


if __name__ == "__main__":
    main()
