#!/usr/bin/env bash
# Build the leakage-free XL eval corpus.
#
# The routing work hit a wall at 64 images: nine coefficient features correlate
# 0.5-0.67 with per-image gain at fixed quality, and none of them beats a single
# binary threshold, because a per-cell linear model is 40 parameters and the
# across-split spread (±0.3 ssim2) swamps the effects being chased (±0.05).
# The constraint is corpus size, not feature choice.
#
# CORPUS REPOINT, 2026-09-07 (docs/CORPUS-REPOINT-HANDOFF.md). Two legs changed
# and one assumption died.
#
# The old header claimed no exclusion list was needed, because training read only
# eight named subcorpora of /mnt/v/imazen-26 and no leg here was one of them.
# That justification is gone: training now reads the TRAIN BUCKET of the whole
# canonical corpus, so any leg that shares content with it must be filtered, not
# waved through. The noaa leg below is filtered for exactly that reason. Run
# tools/leakage_audit.py over the other legs rather than assuming they are clean
# — assuming is what made this comment wrong the first time.
#
#   noaa  — rebuilt from the canonical 5300-noaa-hurricane-documents, restricted
#           to val+test via eval_split/imazen26_eval_files.tsv. Unfiltered it
#           would put 22 training documents into an eval corpus.
#   patents — filtered. MEASURED 2026-09-07: the canonical corpus's 113 patent
#           scans were drawn from this very corpus, and three of its documents
#           (US5046022, US77494, US3807657 = 39 of 357 pages) are now training
#           data. Excluded per DOCUMENT, not per page: pages of one patent share
#           a scanner, typeface and paper stock, so holding out page 3 of a
#           document whose page 2 was trained on holds out nothing.
#           The membership test is the patent number against CORPUS-MANIFEST.tsv,
#           NOT the perceptual fingerprint. tools/leakage_audit.py flagged 29
#           pages across SIX documents; three of those documents were false
#           positives — near-blank text pages of unrelated patents colliding at
#           16x16 (distances 2.4-2.7, against 0.00 for the true matches, which
#           also agreed by filename). Believing the fingerprint would have thrown
#           away 87 clean pages. On document scans it over-flags; use it to find
#           candidates, then confirm by identity.
#   nasa  — DROPPED, no canonical replacement. Its 24 files came from the deleted
#           acquisition root. The canonical corpus does contain 72 rows matching
#           "nasa", but they are `8100-lilith-web-screenshots` PNGs of nasa.gov
#           pages (source `various-web`, names like `nasa-artemis_dpr1_page1`) —
#           screenshots of a website, not NASA imagery. Substituting them would
#           relabel a content class, not restore a leg.
#
# PNG SOURCES ONLY. JPEG-sourced references penalise the model for removing
# artifacts that are present in the reference (ROADMAP §0.1), which is the
# contamination this corpus exists to avoid. That excludes sierra (598
# e-commerce shots, all JPEG) and art-cc0 — both worth adding later via the
# downscale-to-pristine treatment, which is a separate job.
#
# Kodak is excluded on purpose: banned as overfit by every codec
# (claudehints/topics/benchmarking.md).
set -euo pipefail

# NOT under /mnt/v/imazen* — every path matching that glob is an invalid or
# stale imazen-26 (user directive 2026-09-07), and an eval corpus living there
# invites the next session to mistake it for one.
OUT="${1:-/mnt/v/zensr/xl-eval-corpus}"
mkdir -p "$OUT"

# label<TAB>source directory. Labels group by CONTENT CLASS where the source is
# homogeneous, because the content-split curves are fit per class and the
# grouping has to mean something.
# The canonical corpus REPOSITORY (github.com/imazen/imazen-26). Not
# ~/work/codec-corpus/imazen-26, which is the pre-2026-08-23 location.
CANON="${IMAZEN26_REPO:-$HOME/work/imazen-26}"
MANIFEST="$CANON/CORPUS-MANIFEST.tsv"
# The EFFECTIVE split, not the repo's raw buckets: the near-duplicate
# same-bucketing moves 180 files across the held-out boundary, so using the raw
# buckets here would put training documents into this eval corpus.
SPLIT="${ZENSR_EFFECTIVE_SPLIT:-eval_split/imazen26_effective_split.tsv}"
[ -f "$MANIFEST" ] || { echo "canonical imazen-26 not at $CANON — clone github.com/imazen/imazen-26 or set IMAZEN26_REPO" >&2; exit 1; }
[ -f "$SPLIT" ] || { echo "no $SPLIT — run \`just split\`" >&2; exit 1; }
SOURCES=$(cat <<EOF
patents	/mnt/v/collections/patent-corpus
sci-figures	/mnt/v/collections/sci-figures-color
cid22	/mnt/v/work/codec-corpus/CID22
clic2025	/mnt/v/work/codec-corpus/clic2025
gb82	/mnt/v/work/codec-corpus/gb82
gb82-sc	/mnt/v/work/codec-corpus/gb82-sc
EOF
)

: > "$OUT/SUBCORPORA.tsv"
{
  echo "# XL eval corpus — PNG-only, none of it in any training set."
  echo "# Built by tools/build_xl_corpus.sh; see that file for provenance."
} >> "$OUT/SUBCORPORA.tsv"

total=0
while IFS=$'\t' read -r label src; do
  [ -z "${label:-}" ] && continue
  case "$label" in \#*) continue ;; esac   # tolerate comments in the list
  # Clear the label first. `ln -sf` overwrites but never REMOVES, so without
  # this a file that a later filter excludes stays behind from an earlier build
  # — which is exactly how 6 training images survived the patents/noaa filters
  # into an audited corpus on 2026-09-08.
  rm -rf "$OUT/$label"
  mkdir -p "$OUT/$label"
  n=0
  skipped=0
  while IFS= read -r f; do
    # patents: skip any page of a patent document that is in the canonical
    # corpus (see the header). Document-level, keyed on the patent number.
    if [ "$label" = patents ]; then
      pn=$(basename "$(dirname "$f")")
      if grep -qi -- "$pn" "$MANIFEST" 2>/dev/null; then
        skipped=$((skipped+1)); continue
      fi
    fi
    # Flatten nested layouts into one directory per label, keeping enough of the
    # path to stay unique — several sources nest by document or shard and the
    # basenames collide.
    rel="${f#$src/}"
    flat="${rel//\//__}"
    ln -sf "$f" "$OUT/$label/$flat"
    n=$((n+1))
  done < <(find -L "$src" -type f -iname '*.png' 2>/dev/null | sort)
  if [ "$n" -gt 0 ]; then
    printf '%s\t%s\n' "$label" "$label" >> "$OUT/SUBCORPORA.tsv"
    if [ "$skipped" -gt 0 ]; then
      printf '%-14s %5d  (%d pages excluded: in the canonical corpus)\n' \
        "$label" "$n" "$skipped"
    else
      printf '%-14s %5d\n' "$label" "$n"
    fi
    total=$((total+n))
  fi
done <<< "$SOURCES"

# noaa, held-out only, straight from the corpus repo's canonical split
# (validate + test). Symlinked one file at a time rather than by a find over the
# folder, so the filter cannot be silently bypassed by a later edit that
# "simplifies" it back into the SOURCES table.
rm -rf "$OUT/noaa"
mkdir -p "$OUT/noaa"
n=0
while IFS=$'\t' read -r path; do
  case "$path" in 5300-noaa-hurricane-documents/*) ;; *) continue ;; esac
  src="$CANON/$path"
  [ -f "$src" ] || continue
  flat="${path#5300-noaa-hurricane-documents/}"
  ln -sf "$src" "$OUT/noaa/${flat//\//__}"
  n=$((n+1))
done < <(awk -F'\t' '!/^#/ && $2 != "train" {print $1}' "$SPLIT")
if [ "$n" -gt 0 ]; then
  printf '%s\t%s\n' noaa noaa >> "$OUT/SUBCORPORA.tsv"
  printf '%-14s %5d  (held-out only, of 44)\n' noaa "$n"
  total=$((total+n))
fi

# The marker zensr-bench's resolve_pinned() looks for. Written HERE, by the
# builder, so its claim cannot drift from what was actually built — the previous
# hand-maintained version justified itself with "training reads only these eight
# subcorpora of /mnt/v/imazen-26", which stopped being true the moment the corpus
# was repointed, and nothing forced anyone to notice.
cat > "$OUT/NO_PIN_REQUIRED" <<EOF
this corpus is filtered at BUILD time, so it contains no training images to exclude
Built $(date -u +%Y-%m-%dT%H:%M:%SZ) by tools/build_xl_corpus.sh.
Training is the TRAIN bucket of the canonical corpus repo (github.com/imazen/imazen-26),
manifests/train.tsv. Two legs here overlap it and are filtered:
  patents — documents present in CORPUS-MANIFEST.tsv are dropped whole
            (US5046022, US77494, US3807657 = 39 of 357 pages).
  noaa    — held-out buckets only, via eval_split/imazen26_effective_split.tsv
            (regenerate with: just split), the same file the trainer excludes by.
The other legs (sci-figures, cid22, clic2025, gb82, gb82-sc) were audited clean by
tools/leakage_audit.py on 2026-09-08: 0 flagged files. Re-run it after any retrain,
because the training set is what changes, not this corpus.

KNOWN RESIDUAL, 5 of 828 files, measured 2026-09-08 and left in deliberately:
  3 patent pages (US299894, US3063966, US3819587) are fingerprint FALSE
    POSITIVES — near-blank text pages colliding at 16x16 with pages of an
    unrelated patent (distance 2.4-2.7 against 0.00 for true matches). None of
    those three patents appears in CORPUS-MANIFEST.tsv, which is the deciding
    test; the fingerprint is not.
  2 noaa pages (5330 kirk_p05, 5343 rafael_p19) match TRAINING pages of DIFFERENT
    storms at distance 0.89 and 2.36. NOAA advisories are template-driven, so
    every held-out page resembles training pages of other advisories. This is a
    property of the leg, not a filter bug: removing it means dropping noaa
    entirely. Read noaa numbers as mildly optimistic.
EOF

printf '%-14s %5d\n' TOTAL "$total"
echo "wrote $OUT/SUBCORPORA.tsv and $OUT/NO_PIN_REQUIRED"
