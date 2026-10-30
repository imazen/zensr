#!/usr/bin/env python3
"""Sync imazen-26 image bytes for one split bucket into the canonical checkout.

The canonical repo (imazen26_canonical.REPO) holds manifests only; bytes are
public on R2 (ACCESS.md). Files already present (non-empty) are skipped, so
re-running is cheap; the published bytes differ slightly from the manifest
lengths (metadata whitelist rewrite), so size is not compared. R2 refuses
urllib's default User-Agent with 403, hence curl. Prints a URL\\tpath list when --list is
given, for `xargs -P`.

Usage: fetch_corpus_bytes.py [bucket=train] [--list]
"""
import os
import sys
import subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import imazen26_canonical as c  # noqa: E402


def wanted(bucket):
    for r in c.split_rows(bucket):
        dst = c.abspath(r)
        if os.path.isfile(dst) and os.path.getsize(dst) > 0:
            continue
        yield r["raw_url"], dst


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    bucket = args[0] if args else "train"
    todo = list(wanted(bucket))
    if "--list" in sys.argv:
        for url, dst in todo:
            print(f"{url}\t{dst}")
        return
    print(f"{bucket}: {len(todo)} files to fetch", flush=True)
    for i, (url, dst) in enumerate(todo, 1):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        r = subprocess.run(["curl", "-sS", "--fail", "--retry", "3", "-o", dst + ".part", url],
                           capture_output=True, text=True)
        if r.returncode == 0:
            os.replace(dst + ".part", dst)
        else:
            print(f"FAIL {url}: {r.stderr.strip()}", flush=True)
        if i % 50 == 0:
            print(f"{i}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
