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

OUT="${1:-/mnt/v/imazen-26-clean-xl}"
mkdir -p "$OUT"

# label<TAB>source directory. Labels group by CONTENT CLASS where the source is
# homogeneous, because the content-split curves are fit per class and the
# grouping has to mean something.
CANON="${ZENSR_CANONICAL:-$HOME/work/codec-corpus/imazen-26}"
MANIFEST="$CANON/CORPUS-MANIFEST.tsv"
[ -f "$MANIFEST" ] || { echo "no manifest at $MANIFEST" >&2; exit 1; }
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

# noaa, held-out only. Symlinked one file at a time from the pin rather than by
# a find over the folder, so the filter cannot be silently bypassed by a later
# edit that "simplifies" it back into the SOURCES table.
PIN="${ZENSR_EVAL_PIN:-eval_split/imazen26_eval_files.tsv}"
if [ -f "$PIN" ]; then
  mkdir -p "$OUT/noaa"
  n=0
  while IFS=$'\t' read -r folder fname; do
    case "$folder" in 5300-noaa-hurricane-documents) ;; *) continue ;; esac
    src="$CANON/$folder/$fname"
    [ -f "$src" ] || continue
    ln -sf "$src" "$OUT/noaa/${fname//\//__}"
    n=$((n+1))
  done < <(grep -v '^#' "$PIN")
  if [ "$n" -gt 0 ]; then
    printf '%s\t%s\n' noaa noaa >> "$OUT/SUBCORPORA.tsv"
    printf '%-14s %5d  (held-out only, of 44)\n' noaa "$n"
    total=$((total+n))
  fi
else
  echo "WARNING: no pin at $PIN — noaa leg SKIPPED rather than risk including" >&2
  echo "         training documents in an eval corpus." >&2
fi

# The marker zensr-bench's resolve_pinned() looks for. Written HERE, by the
# builder, so its claim cannot drift from what was actually built — the previous
# hand-maintained version justified itself with "training reads only these eight
# subcorpora of /mnt/v/imazen-26", which stopped being true the moment the corpus
# was repointed, and nothing forced anyone to notice.
cat > "$OUT/NO_PIN_REQUIRED" <<EOF
this corpus is filtered at BUILD time, so it contains no training images to exclude
Built $(date -u +%Y-%m-%dT%H:%M:%SZ) by tools/build_xl_corpus.sh.
Training is the TRAIN bucket of the canonical corpus (~/work/codec-corpus/imazen-26);
see eval_split/imazen26_split.tsv. Two legs here overlap it and are filtered:
  patents — documents present in CORPUS-MANIFEST.tsv are dropped whole
            (US5046022, US77494, US3807657 = 39 of 357 pages).
  noaa    — held-out buckets only, via eval_split/imazen26_eval_files.tsv.
The other legs (sci-figures, cid22, clic2025, gb82, gb82-sc) were audited clean by
tools/leakage_audit.py on 2026-09-07: 0 flagged files. Re-run it after any retrain,
because the training set is what changes, not this corpus.
EOF

printf '%-14s %5d\n' TOTAL "$total"
echo "wrote $OUT/SUBCORPORA.tsv and $OUT/NO_PIN_REQUIRED"
