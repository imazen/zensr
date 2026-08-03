#!/usr/bin/env bash
# Full-budget affinity-KD pair on the LOCAL box (RTX 5070).
#
# Measured 4,601 steps/min on the matched microbenchmark against 2,478 for an
# idle RTX 2080 — WSL2 is not a tax on CUDA compute. 12 GB of VRAM also keeps
# the 14.7 GB dataset closer to device-resident than the 8 GB cards managed.
#
# Everything heavy goes through run-heavy: nice -n19 ionice -c3, a hard memory
# ceiling, and a capped thread count so the box stays usable. The GPU is the
# bottleneck here, not the CPU, so the CPU cap costs nothing.
set -euo pipefail

W="${1:?affinity weight — calibrate from a probe, do not guess}"
STEPS="${2:-200000}"
BATCH="${3:-48}"
LR="${4:-3e-4}"
DATA="${ZENSR_FKD_DATA:-$HOME/tmp/zensr-big-d7}"
TCKPT="${ZENSR_FKD_TCKPT:-$HOME/tmp/zensr-ckpts/jason/dejpeg11_teacher_100000.pth}"
PY="${ZENSR_PY:-python3}"
JOBS="${ZENSR_JOBS:-8}"
# Both arms share a seed; a DIFFERENT seed from the pair running elsewhere makes
# this an independent replication rather than a duplicate. Comparing across
# seeds (this box's B against another box's A) would confound seed with
# treatment, which is why both arms are run here rather than only the missing one.
SEED_T="${ZENSR_SEED_TORCH:-2}"
SEED_N="${ZENSR_SEED_NUMPY:-3}"

for f in "$DATA/lr_u8.npy" "$TCKPT"; do
  [ -f "$f" ] || { echo "missing: $f" >&2; exit 2; }
done
cd "${ZENSR_REPO:-$HOME/work/zen/zensr}"

run_arm() {
  local name="$1"
  local w="$2"
  local log="$HOME/tmp/fkdlocal_${name}.log"
  echo "== arm $name (ZENSR_FKD_W=$w) -> $log"
  ZENSR_DATA="$DATA" ZENSR_SCALE=1 ZENSR_NF=24 ZENSR_NC=8 ZENSR_QBOOST=3 \
  ZENSR_FKD_TEACHER="$TCKPT" ZENSR_FKD_TNF=64 ZENSR_FKD_TNC=16 \
  ZENSR_FKD_W="$w" ZENSR_OUT="fkdlocal_${name}" ZENSR_CKPT_EVERY=2000 \
  ZENSR_SEED_TORCH="$SEED_T" ZENSR_SEED_NUMPY="$SEED_N" \
    ~/work/zen/scripts/run-heavy --mem 32G --jobs "$JOBS" -- \
    "$PY" tools/train_people.py "$STEPS" "$BATCH" "$LR" > "$log" 2>&1
  echo "== arm $name done: $(tail -1 "$log")"
}

# Control first: if output-KD alone fails, the comparison is void and no GPU
# time goes to the treatment arm.
run_arm a_outkd 0
run_arm b_affinity "$W"
echo DONE > "$HOME/tmp/fkdlocal.done"
