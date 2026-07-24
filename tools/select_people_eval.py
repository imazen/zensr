#!/usr/bin/env python3
"""Select the frozen people eval slice from the harvested pxhere pool.

Deterministic: strong-tier images only, grouped by source shard, round-robin
across shards in sorted image_id order (spreads photographer/session bias),
decode-verified (cv2, shorter side >= 1024). Writes:
  <OUT>/eval/<id>.jpg           (Tower, canonical)
  <OUT>/eval_ids.txt            (the freeze — NEVER train on these ids)
  /mnt/v/input/zensr-people-eval-v1/   (local mirror for fast eval reads)

Usage: select_people_eval.py [n=64]
"""
import os
import shutil
import sys
from collections import defaultdict

import cv2

OUT = os.environ.get("ZENSR_PEOPLE_OUT", "/mnt/tower/input/zensr-people-v1")
LOCAL = os.environ.get("ZENSR_PEOPLE_LOCAL", "/mnt/v/input/zensr-people-eval-v1")


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    rows = []
    with open(os.path.join(OUT, "MANIFEST.tsv")) as f:
        header = f.readline()
        for line in f:
            c = line.rstrip("\n").split("\t")
            if len(c) >= 6 and c[1] == "strong":
                rows.append(c)
    by_shard = defaultdict(list)
    for c in rows:
        by_shard[c[3]].append(c)
    for s in by_shard:
        by_shard[s].sort(key=lambda c: int(c[0]))
    order = sorted(by_shard)
    os.makedirs(os.path.join(OUT, "eval"), exist_ok=True)
    os.makedirs(LOCAL, exist_ok=True)
    picked = []
    idx = 0
    while len(picked) < n and any(by_shard[s] for s in order):
        s = order[idx % len(order)]
        idx += 1
        if not by_shard[s]:
            continue
        c = by_shard[s].pop(0)
        src = os.path.join(OUT, "pool", f"{c[0]}.jpg")
        img = cv2.imread(src, cv2.IMREAD_COLOR)
        if img is None or min(img.shape[:2]) < 1024:
            continue
        picked.append(c)
        shutil.copy2(src, os.path.join(OUT, "eval", f"{c[0]}.jpg"))
        shutil.copy2(src, os.path.join(LOCAL, f"{c[0]}.jpg"))
    with open(os.path.join(OUT, "eval_ids.txt"), "w") as f:
        for c in picked:
            f.write(f"{c[0]}\n")
    man = os.path.join(LOCAL, "MANIFEST.tsv")
    with open(man, "w") as f:
        f.write(header)
        for c in picked:
            f.write("\t".join(c) + "\n")
    print(f"picked {len(picked)} eval images across {len({c[3] for c in picked})} shards")
    print(f"frozen: {os.path.join(OUT, 'eval_ids.txt')}; local mirror: {LOCAL}")


if __name__ == "__main__":
    main()
