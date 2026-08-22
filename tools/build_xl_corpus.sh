#!/usr/bin/env bash
# Build the leakage-free XL eval corpus.
#
# The routing work hit a wall at 64 images: nine coefficient features correlate
# 0.5-0.67 with per-image gain at fixed quality, and none of them beats a single
# binary threshold, because a per-cell linear model is 40 parameters and the
# across-split spread (±0.3 ssim2) swamps the effects being chased (±0.05).
# The constraint is corpus size, not feature choice.
#
# Every source below is verifiably absent from training. Training reads only
# `/mnt/v/imazen-26` and only these eight subcorpora (make_distill_data.py:27):
# lilith, unsplash-people, screen, internet-archive-scans, national-park-service,
# unsplash-renders, unsplash-textures, office-documents. Nothing here is one of
# them, so no pinned-exclusion list is needed — every file is eval-valid.
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
# INVALID ROOT WARNING: the nasa and noaa legs below come from
# /mnt/v/imazen-26, the pre-curation corpus — 68 of this corpus's 913 files.
# The only valid imazen-26 is ~/work/codec-corpus/imazen-26 (user directive
# 2026-08-05, benchmarks/imazen26_contamination_audit_2026-08-05.md). Rebuild
# those two from 5300-noaa-hurricane-documents and the nasa equivalent when the
# corpus is re-derived.
SOURCES=$(cat <<'EOF'
patents	/mnt/v/collections/patent-corpus
sci-figures	/mnt/v/collections/sci-figures-color
cid22	/mnt/v/work/codec-corpus/CID22
clic2025	/mnt/v/work/codec-corpus/clic2025
gb82	/mnt/v/work/codec-corpus/gb82
gb82-sc	/mnt/v/work/codec-corpus/gb82-sc
nasa	/mnt/v/imazen-26/nasa
noaa	/mnt/v/imazen-26/noaa
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
  while IFS= read -r f; do
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
    printf '%-14s %5d\n' "$label" "$n"
    total=$((total+n))
  fi
done <<< "$SOURCES"

printf '%-14s %5d\n' TOTAL "$total"
echo "wrote $OUT/SUBCORPORA.tsv"
