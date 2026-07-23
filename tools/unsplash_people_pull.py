#!/usr/bin/env python3
"""Supplemental people photos from Unsplash (user-authorized 2026-07-24).

STATUS 2026-07-24: the keyless napi endpoint returns 401 for search — Unsplash
locked it down. This script is KEPT for when an official API key exists
(swap the napi URL for api.unsplash.com + Authorization: Client-ID header).
Result of the attempt: kept=0; pxhere CC0 pool carries the corpus.

Uses the site's public napi search endpoint (keyless) at a polite rate
(1 req/s, ~300 photos). TRAIN-POOL ONLY — the redistributable eval slice
stays pxhere-CC0 (Unsplash License permits commercial use but is not CC0;
its anti-replication clause makes benchmark redistribution gray).

Writes <OUT>/pool-unsplash/<id>.jpg + MANIFEST-unsplash.tsv with photo page
URL, photographer name + profile URL, dims, query. Skips duplicates.

Usage: unsplash_people_pull.py [target=300]
"""
import json
import os
import sys
import time
import urllib.request

OUT = os.environ.get("ZENSR_PEOPLE_OUT", "/mnt/tower/input/zensr-people-v1")
QUERIES = [
    "portrait", "people", "family", "candid street people", "child portrait",
    "man portrait", "woman portrait", "elderly person", "dark skin portrait",
    "black woman portrait", "asian man portrait", "latina portrait",
    "indian family", "african family", "wedding couple", "athlete action",
    "musician performing", "crowd festival", "worker hands face",
]
UA = {"User-Agent": "Mozilla/5.0 (zensr corpus builder; contact: imazen.io)"}


def get_json(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def main():
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    pool = os.path.join(OUT, "pool-unsplash")
    os.makedirs(pool, exist_ok=True)
    man = os.path.join(OUT, "MANIFEST-unsplash.tsv")
    seen = {os.path.splitext(f)[0] for f in os.listdir(pool) if f.endswith(".jpg")}
    if not os.path.exists(man):
        with open(man, "w") as f:
            f.write("id\twidth\theight\tquery\tphoto_url\tphotographer\tprofile_url\tlicense\n")
    kept = len(seen)
    per_q = max(2, target // len(QUERIES) // 20 + 1)  # pages per query
    for q in QUERIES:
        if kept >= target:
            break
        for page in range(1, per_q + 1):
            if kept >= target:
                break
            url = (f"https://unsplash.com/napi/search/photos?query="
                   f"{urllib.request.quote(q)}&page={page}&per_page=20")
            try:
                d = get_json(url)
            except Exception as e:
                print(f"query '{q}' p{page}: {e}", flush=True)
                time.sleep(5)
                continue
            time.sleep(1.0)
            for ph in d.get("results", []):
                if kept >= target:
                    break
                pid = ph["id"]
                if pid in seen or ph.get("premium") or ph.get("plus"):
                    continue
                w, h = ph.get("width", 0), ph.get("height", 0)
                if min(w, h) < 1200:
                    continue
                raw = ph.get("urls", {}).get("raw")
                if not raw:
                    continue
                dl = f"{raw.split('?')[0]}?q=90&w=2400&fm=jpg"
                try:
                    req = urllib.request.Request(dl, headers=UA)
                    with urllib.request.urlopen(req, timeout=60) as r:
                        data = r.read()
                except Exception as e:
                    print(f"{pid}: {e}", flush=True)
                    continue
                time.sleep(1.0)
                if len(data) < 100_000:
                    continue
                with open(os.path.join(pool, f"{pid}.jpg"), "wb") as f:
                    f.write(data)
                user = ph.get("user", {})
                with open(man, "a") as f:
                    f.write(f"{pid}\t{w}\t{h}\t{q}\t"
                            f"https://unsplash.com/photos/{pid}\t"
                            f"{user.get('name', '?')}\t"
                            f"{user.get('links', {}).get('html', '?')}\t"
                            f"Unsplash License\n")
                seen.add(pid)
                kept += 1
            print(f"'{q}' p{page}: kept so far {kept}", flush=True)
    print(f"UNSPLASH DONE kept={kept}", flush=True)


if __name__ == "__main__":
    main()
