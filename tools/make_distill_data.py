#!/usr/bin/env python3
"""Generate distillation pairs for the realtime-2x student (S-E pilot).

Input crops: HR from the **canonical** imazen-26 corpus repo
(github.com/imazen/imazen-26, checked out at `~/work/imazen-26`), restricted to
the TRAIN bucket of its canonical split, downscaled 2x (area) then JPEG-degraded
via cv2 (libjpeg-turbo lineage) at q in [40,90], 4:2:0.
Target: 2xNomosUni_span_multijpg (teacher) output on the degraded LR, computed
on GPU with the same functional forward as dump_adopted.py (merged Conv3XC).

Output shards: ~/tmp/zensr-distill/{lr_u8.npy, teacher_f16.npy, meta.json}
(lr 96x96 u8 HWC, teacher 192x192 f16 CHW). Val split = last 512 pairs.

**Repointed 2026-09-07/08** from `/mnt/v/imazen-26` (the pre-curation
acquisition corpus, since deleted) to the canonical corpus REPOSITORY, per the
user directive and `docs/CORPUS-REPOINT-HANDOFF.md`. Note the destination is the
repo, not `~/work/codec-corpus/imazen-26` — that is the pre-2026-08-23 location
and is stale. Two things changed together:

* the root and the subcorpus names (`tools/imazen26_canonical.py` holds the
  mapping — it is not 1:1, and one old subcorpus has no canonical equivalent);
* **eval exclusion is now by split bucket, not by "first 8 sorted ∪ a pinned
  list"**. That old scheme leaked twice. The split is not ours to invent: it is
  the repo's `manifests/split_map.tsv`, read by `tools/corpus_split.py`, which
  adds only the near-duplicate same-bucketing the repo itself prescribes.

Anything trained before this date was fitted through the wrong corpus and is
provisional.
"""
import json
import os
import random
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from corpus_split import split_map  # noqa: E402
from dump_adopted import compact_forward, load_sd, prepare_span_sd, span_forward  # noqa: E402
from imazen26_canonical import REPO, all_folders, canonical_for  # noqa: E402

ROOT = os.environ.get("ZENSR_ROOT", REPO)
# Default: every canonical folder. The old default named eight flat subcorpora
# that no longer exist; restricting the new corpus to their equivalents would
# preserve a limitation that only ever existed because the root was wrong, and
# would throw away ~1,257 images of content the model has never seen (AI
# products and clipart, plots, patent scans, museum photography).
SUBS = all_folders()
# ZENSR_SUBS restricts sources (class-specialist datasets, e.g. the graphics
# model). Accepts canonical folder names AND the old flat names, which are
# translated through the mapping — an old name with no canonical home raises
# rather than silently contributing zero files.
if os.environ.get("ZENSR_SUBS"):
    want = []
    for s in os.environ["ZENSR_SUBS"].split(","):
        s = s.strip()
        if not s:
            continue
        want += [s] if s in SUBS else canonical_for(s)
    SUBS = want
OUT = os.path.expanduser(os.environ.get("ZENSR_DATA", "~/tmp/zensr-distill"))
N_PAIRS = 14000
CROP = 192  # HR crop; LR = 96
SEED = 20260723

SPLIT = split_map()


def list_train_files(sub):
    """Canonical-corpus files in folder `sub` whose split bucket is train.

    No first-N rule and no pinned list: the bucket IS the exclusion. Val and
    test are held out by construction, and every derivative of an origin
    inherits the origin's bucket, so nothing crosses the split.
    """
    files = sorted(os.path.join(ROOT, p) for p, b in SPLIT.items()
                   if b == "train" and p.split("/", 1)[0] == sub)
    # ZENSR_CLEAN_GT=1: drop JPEG-sourced references. They are themselves
    # compressed, so training on them teaches the model to REPRODUCE jpeg
    # artifacts (the training-side twin of the 2026-07-31 eval contamination).
    #
    # WARNING on the canonical corpus: this is no longer a light filter. 20% of
    # the corpus is JPEG and 4% HEIC, concentrated in exactly the photographic
    # folders — 2000-unsplash-people is 28/28 JPEG, 1600-lilith-food 39/41.
    # Enabling this drops nearly all photographic content and leaves a corpus of
    # screenshots, plots, AI renders and scans, which would quietly turn the
    # "photo" leg of the content-split curves into a fiction. The real fix is the
    # downscale-to-pristine treatment (handoff §5 step 2), not this flag.
    if os.environ.get("ZENSR_CLEAN_GT") == "1":
        files = [f for f in files if f.lower().endswith(".png")]
    return files


# ZENSR_FOLDER_CAP=<fraction>: no single canonical folder may contribute more
# than this share of training files. Unset = no cap.
#
# Why it exists: the canonical corpus is not balanced for this task.
# `9226-lilith-ai-products` alone is 33% of the training crops and the corpus is
# ~68% synthetic/graphic against ~18% photographic — while zensr restores web
# JPEGs, which are mostly photographs. Nobody chose that mixture; it falls out of
# "train on the whole corpus", which is the only defensible default once the old
# eight-subcorpus list is gone. A cap makes the choice explicit and recordable.
# 0.15 takes ai-products from 33% to 15% and leaves every other folder untouched,
# at a cost of ~18% of the pool.
#
# NOT applied by default: changing the training mixture is a decision, and the
# first ladder on the new corpus should measure the uncapped mixture so the cap
# has something to be compared against.
FOLDER_CAP = float(os.environ.get("ZENSR_FOLDER_CAP", "0") or 0)


def cap_folders(per_folder, rng):
    """Flatten `{folder: [file]}` into one pool, optionally capping each folder's
    share at FOLDER_CAP.

    The cap is on each folder's share of the FINAL pool, which needs a fixed
    point: trimming the biggest folder shrinks the pool, which raises everyone
    else's share, which can push the next folder over. Capping against the
    original total instead is the obvious shortcut and it does not hold — at 15%
    it leaves `8100-lilith-web-screenshots` at 20% of what remains.

    Sampling within an over-cap folder is random under the run seed rather than a
    prefix, so the cap does not silently select by filename.
    """
    sizes = {k: len(v) for k, v in per_folder.items()}
    if 0 < FOLDER_CAP < 1:
        for _ in range(100):
            total = sum(sizes.values())
            over = {k: n for k, n in sizes.items() if n > FOLDER_CAP * total}
            if not over:
                break
            # Trim the single worst offender per round; trimming all at once
            # overshoots, because each trim lowers the bar for the others.
            worst = max(over, key=lambda k: sizes[k])
            # n / (total - sizes[worst] + n) <= cap  ->  solve for n
            rest = total - sizes[worst]
            sizes[worst] = max(1, int(FOLDER_CAP * rest / (1 - FOLDER_CAP)))
    pool = []
    for folder in sorted(per_folder):
        fs = sorted(per_folder[folder])
        kept = fs if sizes[folder] >= len(fs) else sorted(rng.sample(fs, sizes[folder]))
        note = f"  (capped from {len(fs)})" if len(kept) != len(fs) else ""
        print(f"{folder}: {len(kept)} train files{note}")
        pool += kept
    if 0 < FOLDER_CAP < 1:
        n = sum(len(v) for v in per_folder.values())
        print(f"folder cap {FOLDER_CAP:.0%}: pool {n} -> {len(pool)}")
    return pool


def ref_provenance(files):
    """Count references by kind. A JPEG ground truth is itself compressed, so a
    pair built from one measures artifact REPRODUCTION as fidelity; the ladder
    has to be reported split by this."""
    out = {"png": 0, "jpg": 0, "heic": 0, "other": 0}
    for f in files:
        e = f.rsplit(".", 1)[-1].lower()
        out["jpg" if e in ("jpg", "jpeg") else
            "heic" if e in ("heic", "heif") else
            "png" if e == "png" else "other"] += 1
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = random.Random(SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # Teacher selection (ZENSR_TEACHER): "span" = 2xNomosUni_span (SSIM2 king at
    # q<=50), "compact" = 2xNomosUni_compact (q75 + butteraugli + worse-rate king).
    # prepare_span_sd merges Conv3XC branches; span_forward normalizes input
    # itself ((x-mean)*255, official). The first 14k-pair run predated the norm
    # + inplace-SiLU concat fixes -> constant-gray teacher, fully discarded.
    W = "/mnt/tower/output/zensr-training/adopted-weights"
    teacher = os.environ.get("ZENSR_TEACHER", "span")
    if teacher == "compact":
        sd = load_sd(os.path.join(W, "2xNomosUni_compact_multijpg_ldl.pth"))
        fwd = lambda t: compact_forward(sd, t, 2)
    else:
        sd, _ = prepare_span_sd(os.path.join(W, "2xNomosUni_span_multijpg.pth"))
        fwd = lambda t: span_forward(sd, t, 2)
    sd = {k: v.to(dev) for k, v in sd.items()}

    per_folder = {}
    for s in SUBS:
        fs = list_train_files(s)
        if fs:
            per_folder[s] = fs
    pool = cap_folders(per_folder, rng)
    prov = ref_provenance(pool)
    print(f"reference provenance: {prov}")
    if not pool:
        sys.exit("no training files — check ZENSR_ROOT / ZENSR_SUBS")
    rng.shuffle(pool)
    # image-level val: last 512 pairs come ONLY from val-reserved files
    n_val_files = max(16, len(pool) // 20)
    val_pool, train_pool = pool[-n_val_files:], pool[:-n_val_files]
    TRAIN_TARGET = N_PAIRS - 512
    print(f"train files {len(train_pool)} / val files {n_val_files}", flush=True)

    lr_all = np.zeros((N_PAIRS, 96, 96, 3), dtype=np.uint8)
    tg_all = np.zeros((N_PAIRS, 3, 192, 192), dtype=np.float16)
    made = 0
    fi = 0
    batch_lr = []
    while made < N_PAIRS:
        phase_target = TRAIN_TARGET if made < TRAIN_TARGET else N_PAIRS
        src = train_pool if made < TRAIN_TARGET else val_pool
        f = src[fi % len(src)]
        fi += 1
        img = cv2.imread(f, cv2.IMREAD_COLOR)  # BGR
        if img is None or img.shape[0] < CROP or img.shape[1] < CROP:
            continue
        for _ in range(min(4, 1 + img.shape[0] * img.shape[1] // (CROP * CROP * 4))):
            if made + len(batch_lr) >= phase_target:
                break
            y = rng.randrange(0, img.shape[0] - CROP + 1)
            x = rng.randrange(0, img.shape[1] - CROP + 1)
            hr = img[y:y + CROP, x:x + CROP]
            lr = cv2.resize(hr, (96, 96), interpolation=cv2.INTER_AREA)
            q = rng.randrange(40, 91)
            ok, enc = cv2.imencode(".jpg", lr, [cv2.IMWRITE_JPEG_QUALITY, q])
            if not ok:
                continue
            lr = cv2.imdecode(enc, cv2.IMREAD_COLOR)
            batch_lr.append(lr)
        if len(batch_lr) >= 32 or (made + len(batch_lr) >= phase_target and batch_lr):
            arr = np.stack(batch_lr)  # B,96,96,3 BGR u8
            rgb = arr[..., ::-1].astype(np.float32) / 255.0
            t = torch.from_numpy(rgb.transpose(0, 3, 1, 2).copy()).to(dev)
            with torch.no_grad():
                out = fwd(t).clamp(0, 1)
            n = len(batch_lr)
            lr_all[made:made + n] = arr[..., ::-1]  # store RGB u8
            tg_all[made:made + n] = out.cpu().numpy().astype(np.float16)
            made += n
            batch_lr = []
            if made % 1024 < 32:
                print(f"{made}/{N_PAIRS}")
    np.save(os.path.join(OUT, "lr_u8.npy"), lr_all)
    np.save(os.path.join(OUT, "teacher_f16.npy"), tg_all)
    json.dump({"n": N_PAIRS, "val_tail": 512, "teacher": "2xNomosUni_span_multijpg",
               "degrade": "area-down2x + cv2 jpeg q40-90", "seed": SEED,
               "corpus": ROOT,
               "corpus_folders": SUBS,
               "eval_split_excluded": "origin-level split, bucket != train "
                                      "(eval_split/imazen26_split.tsv)",
               "clean_gt_filter": os.environ.get("ZENSR_CLEAN_GT") == "1",
               # Per handoff §5: record what KIND of reference each pair came
               # from, because the 2026-07 defect happened for want of this column.
               "ref_provenance": ref_provenance(train_pool + val_pool),
               "val_split": "image-level (last 5% of shuffled files)"},
              open(os.path.join(OUT, "meta.json"), "w"), indent=1)
    print("DONE", lr_all.nbytes / 1e9, "GB +", tg_all.nbytes / 1e9, "GB")


if __name__ == "__main__":
    main()
