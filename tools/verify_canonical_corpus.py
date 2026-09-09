#!/usr/bin/env python3
"""Is the local canonical-corpus checkout actually the published corpus?

**Verifying against the manifest is not enough, and this is not hypothetical.**
On 2026-09-09 a full 2,160-file manifest verification of the checkout reported 0
mismatches while `5300-noaa-hurricane-documents/5314_noaa_nhc-al062024-francine_p01_2550x3300.png`
was 54.4% NUL and would not decode. Its manifest `sha256` is the hash of the
corrupt bytes: the file was damaged (one fully-zeroed 1 MiB-aligned block, length
unchanged at 1,937,400) *before* the manifest was generated, so the manifest
certified the damage. imazen/imazen-26#2.

The object store is the independent witness. R2 returns a plain MD5 as the ETag
for single-part uploads, so most of the corpus verifies with a HEAD request; only
the handful of multipart objects need a full GET.

Three tests, cheapest first:
  1. ETag (MD5) vs local — 2,151 of 2,160 files, ~40 s at 8-way concurrency.
  2. Full GET vs local sha256 — the 6 multipart objects, 325 MB. Skipped unless
     --multipart, because it is the only slow part.
  3. Decode + MiB-aligned NUL-block scan — catches damage that would also have
     been hashed into a future manifest, and needs no network at all.

Gotcha worth keeping: `codec-corpus.r2.imazen.org` returns **403** to a default
`python-urllib/3.x` user-agent. Without a normal UA every file reads as
divergent, which looks like corpus-wide corruption rather than a UA filter — it
did here, for one run.

Usage:
  verify_canonical_corpus.py              # ETag + local decode/NUL scan
  verify_canonical_corpus.py --multipart  # also full-GET the multipart objects
  verify_canonical_corpus.py --offline    # local decode/NUL scan only
"""
import argparse
import csv
import hashlib
import os
import sys
import urllib.request
import warnings
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings("ignore")
from PIL import Image  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from imazen26_canonical import REPO  # noqa: E402

UA = {"User-Agent": "curl/8.5.0"}
MIB = 1 << 20
ZERO = bytes(MIB)
# PIL has no decoder for these; absence of a decode error means nothing for them,
# so they are reported as unchecked rather than counted as clean.
NO_DECODER = (".heic", ".heif", ".dng")


def manifest_rows():
    rows = []
    for b in ("train", "validate", "test"):
        p = os.path.join(REPO, "manifests", f"{b}.tsv")
        with open(p) as f:
            rows += [(b, r) for r in csv.DictReader(f, delimiter="\t")]
    return rows


def check_etag(item):
    b, r = item
    p = os.path.join(REPO, r["path"])
    if not os.path.exists(p):
        return (b, r["path"], "MISSING", "")
    try:
        req = urllib.request.Request(r["raw_url"], method="HEAD", headers=UA)
        with urllib.request.urlopen(req, timeout=60) as resp:
            etag = resp.headers.get("ETag", "").strip('"')
    except Exception as e:
        code = getattr(e, "code", type(e).__name__)
        return (b, r["path"], f"UNREACHABLE({code})", r["raw_url"])
    if "-" in etag:
        return (b, r["path"], "MULTIPART", etag)   # needs --multipart
    md5 = hashlib.md5(open(p, "rb").read()).hexdigest()
    if md5 != etag:
        return (b, r["path"], "DIVERGENT", f"local_md5={md5} r2_etag={etag}")
    return None


def check_multipart(item):
    b, r = item
    h = hashlib.sha256()
    req = urllib.request.Request(r["raw_url"], headers=UA)
    with urllib.request.urlopen(req, timeout=600) as resp:
        while chunk := resp.read(MIB):
            h.update(chunk)
    local = hashlib.sha256(open(os.path.join(REPO, r["path"]), "rb").read()).hexdigest()
    if h.hexdigest() != local:
        return (b, r["path"], "DIVERGENT", f"local={local[:16]} r2={h.hexdigest()[:16]}")
    return None


def check_local(item):
    """Damage detectable without the network: zeroed blocks, or a decode failure."""
    b, r = item
    p = os.path.join(REPO, r["path"])
    if not os.path.exists(p):
        return (b, r["path"], "MISSING", "")
    data = open(p, "rb").read()
    holes = sum(1 for off in range(0, len(data) - MIB + 1, MIB)
                if data[off:off + MIB] == ZERO)
    if holes:
        return (b, r["path"], "NUL-HOLES", f"{holes} zeroed MiB block(s) in {len(data)} bytes")
    if os.path.splitext(p)[1].lower() in NO_DECODER:
        return None
    try:
        Image.open(p).load()
    except Exception as e:
        return (b, r["path"], "UNDECODABLE", type(e).__name__)
    return None


def report(title, bad, total):
    print(f"\n{title}: {total} checked, {len(bad)} flagged")
    for b, p, kind, detail in sorted(bad):
        print(f"  [{b:8}] {kind:16} {p}")
        if detail:
            print(f"            {detail}")
    return len(bad)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--multipart", action="store_true",
                    help="full-GET the multipart objects (325 MB) instead of skipping them")
    ap.add_argument("--offline", action="store_true",
                    help="local decode + NUL-block scan only")
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()

    rows = manifest_rows()
    print(f"checkout: {REPO}\nmanifest: {len(rows)} files")
    fatal = 0

    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        local_bad = [x for x in ex.map(check_local, rows) if x]
    fatal += report("local scan (NUL holes / decode)", local_bad, len(rows))
    n_nodec = sum(1 for _, r in rows
                  if os.path.splitext(r["path"])[1].lower() in NO_DECODER)
    print(f"  ({n_nodec} heic/dng not decode-checked — no PIL decoder)")

    if a.offline:
        return 1 if fatal else 0

    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        etag_bad = [x for x in ex.map(check_etag, rows) if x]
    mp = [(b, r) for b, r in rows
          if any(p == r["path"] and k == "MULTIPART" for _, p, k, _ in etag_bad)]
    hard = [x for x in etag_bad if x[2] not in ("MULTIPART",)]
    fatal += report("R2 ETag (MD5)", hard, len(rows))
    print(f"  ({len(mp)} multipart objects deferred — rerun with --multipart)")

    if a.multipart and mp:
        with ThreadPoolExecutor(max_workers=4) as ex:
            mp_bad = [x for x in ex.map(check_multipart, mp) if x]
        fatal += report("multipart full-GET", mp_bad, len(mp))

    print(f"\n{'FAIL' if fatal else 'OK'}: {fatal} file(s) flagged")
    return 1 if fatal else 0


if __name__ == "__main__":
    sys.exit(main())
