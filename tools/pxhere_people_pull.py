#!/usr/bin/env python3
"""Harvest CC0 people photos from nyuuzyou/pxhere (HF WebDataset shards).

pxhere.com photos are CC0; the HF dump carries per-image JSON with `tags`,
`download_url`, and `exif_info.resolution` — filterable without decoding.
Streams a seeded sample of shards, keeps people-tagged photos >= MIN_SIDE,
writes <id>.jpg + <id>.json + a provenance MANIFEST.tsv (URLs preserved).

Resumable: finished shards are recorded in _done/. Quota-capped.

Usage: pxhere_people_pull.py [n_shards=40] [quota=2500]
Env:   ZENSR_PEOPLE_OUT (default /mnt/tower/input/zensr-people-v1)
"""
import io
import json
import os
import random
import subprocess
import sys
import tarfile

OUT = os.environ.get("ZENSR_PEOPLE_OUT", "/mnt/tower/input/zensr-people-v1")
BASE = "https://huggingface.co/datasets/nyuuzyou/pxhere/resolve/main"
N_SHARDS_TOTAL = 1101  # pxhere-000000 .. pxhere-001100

# Tier 1: face/person subjects (eval-eligible). Tier 2: skin/body-part closeups
# (training-eligible). Tag match is exact on lowercased tag strings.
STRONG = {
    "portrait", "man", "woman", "girl", "boy", "child", "face", "people",
    "person", "family", "baby", "smile", "couple", "bride", "groom", "selfie",
    "toddler", "senior", "grandmother", "grandfather", "musician", "athlete",
    "dancer", "student", "crowd", "children", "men", "women", "human face",
}
PARTS = {"hand", "finger", "arm", "leg", "skin", "hands", "feet", "foot"}
MIN_SIDE = 1200
MAX_SIDE = 9000


def classify(tags):
    t = {x.strip().lower() for x in tags}
    if t & STRONG:
        return "strong"
    if t & PARTS:
        return "parts"
    return None


def res_ok(meta):
    r = meta.get("exif_info", {}).get("resolution", "")
    try:
        w, h = (int(x) for x in r.lower().split("x"))
    except ValueError:
        return False
    return MIN_SIDE <= min(w, h) and max(w, h) <= MAX_SIDE


def main():
    n_shards = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    quota = int(sys.argv[2]) if len(sys.argv) > 2 else 2500
    os.makedirs(os.path.join(OUT, "pool"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "_done"), exist_ok=True)
    man_path = os.path.join(OUT, "MANIFEST.tsv")
    if not os.path.exists(man_path):
        with open(man_path, "w") as f:
            f.write("image_id\ttier\tresolution\tshard\tdownload_url\ttags\n")
    kept_total = sum(1 for f in os.listdir(os.path.join(OUT, "pool")) if f.endswith(".jpg"))
    rng = random.Random(int(os.environ.get("ZENSR_SHARD_SEED", "20260723")))
    avoid = set()
    ad = os.environ.get("ZENSR_AVOID_DONE", "")
    if ad and os.path.isdir(ad):
        avoid = {int(n.split("-")[1]) for n in os.listdir(ad) if n.startswith("pxhere-")}
    cand = [i for i in range(N_SHARDS_TOTAL) if i not in avoid]
    shards = rng.sample(cand, n_shards)
    print(f"avoiding {len(avoid)} already-used shards", flush=True)
    print(f"resuming with {kept_total} kept; shard plan: {shards}", flush=True)
    for si in shards:
        if kept_total >= quota:
            print(f"quota {quota} reached", flush=True)
            break
        name = f"pxhere-{si:06d}"
        done_mark = os.path.join(OUT, "_done", name)
        if os.path.exists(done_mark):
            continue
        url = f"{BASE}/{name}.tar"
        print(f"streaming {name} (kept so far {kept_total})", flush=True)
        proc = subprocess.Popen(
            ["curl", "-sL", "--fail", url], stdout=subprocess.PIPE, bufsize=1 << 20
        )
        kept_shard = 0
        try:
            t = tarfile.open(fileobj=proc.stdout, mode="r|")
            pending = {}  # id -> (jpg_bytes or meta) pairing within stream
            for m in t:
                base, ext = os.path.splitext(os.path.basename(m.name))
                if ext not in (".json", ".jpg", ".jpeg"):
                    continue
                data = t.extractfile(m).read()
                slot = pending.setdefault(base, {})
                slot[".json" if ext == ".json" else ".jpg"] = data
                if ".json" in slot and ".jpg" in slot:
                    meta = json.loads(slot[".json"])
                    tier = classify(meta.get("tags", []))
                    if tier and res_ok(meta):
                        iid = meta["image_id"]
                        with open(os.path.join(OUT, "pool", f"{iid}.jpg"), "wb") as f:
                            f.write(slot[".jpg"])
                        with open(os.path.join(OUT, "pool", f"{iid}.json"), "wb") as f:
                            f.write(slot[".json"])
                        with open(man_path, "a") as f:
                            f.write(
                                f"{iid}\t{tier}\t{meta['exif_info']['resolution']}\t{name}\t"
                                f"{meta['download_url']}\t{','.join(meta.get('tags', []))[:400]}\n"
                            )
                        kept_total += 1
                        kept_shard += 1
                        if kept_total >= quota:
                            break
                    del pending[base]
        except tarfile.ReadError as e:
            print(f"{name}: tar ended ({e})", flush=True)
        finally:
            proc.stdout.close()
            proc.wait()
        with open(done_mark, "w") as f:
            f.write(str(kept_shard))
        print(f"{name}: kept {kept_shard}", flush=True)
    print(f"HARVEST DONE kept_total={kept_total}", flush=True)


if __name__ == "__main__":
    main()
