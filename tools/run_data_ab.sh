#!/usr/bin/env bash
# Controlled test of the training-data scale-up (ROADMAP 1.2).
#
# The two earlier attempts were confounded — the student run was warm-restarted
# mid-way, and the two teachers differed in init AND AMP dtype as well as
# dataset size. This varies ONE thing: ZENSR_DATA. Same architecture, same
# steps, same batch, same lr, same qboost, same seeds (fixed in the trainer),
# same box (so the same AMP dtype), both from scratch (no ZENSR_INIT), run
# sequentially so neither contends for the GPU.
#
# Both datasets must already carry precomputed hr_f16 targets from the SAME
# teacher, or the arms differ in their target too — check before running.
#
# Usage: run_data_ab.sh <small-data-dir> <big-data-dir> [steps] [batch] [lr]
set -euo pipefail

SMALL="${1:?dataset dir for the small arm}"
BIG="${2:?dataset dir for the big arm}"
STEPS="${3:-120000}"
BATCH="${4:-48}"
LR="${5:-3e-4}"
PY="${ZENSR_PY:-$HOME/zensr-env/bin/python}"
cd "${ZENSR_REPO:-$HOME/zensr-ablation}"

for d in "$SMALL" "$BIG"; do
  [ -f "$d/lr_u8.npy" ] || { echo "missing $d/lr_u8.npy" >&2; exit 2; }
  [ -f "$d/hr_f16.npy" ] || { echo "missing $d/hr_f16.npy — the arms would differ in target, not just data" >&2; exit 2; }
done

run_arm() {
  local name="$1"
  local data="$2"
  local log="$HOME/dataab_${name}.log"
  echo "== arm $name  data=$data  steps=$STEPS -> $log"
  ZENSR_DATA="$data" ZENSR_SCALE=1 ZENSR_NF=24 ZENSR_NC=8 ZENSR_QBOOST=3 \
  ZENSR_OUT="dataab_${name}" ZENSR_CKPT_EVERY=5000 \
    nice -n 5 "$PY" tools/train_people.py "$STEPS" "$BATCH" "$LR" > "$log" 2>&1
  echo "== arm $name done: $(tail -1 "$log")"
}

run_arm small "$SMALL"
run_arm big "$BIG"
echo "DATA_AB_DONE" > "$HOME/dataab.done"
