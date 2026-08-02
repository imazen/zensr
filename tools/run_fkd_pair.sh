#!/usr/bin/env bash
# Paired feature/affinity-KD rung (ROADMAP 1.4, task #13).
#
# Runs two arms that differ by EXACTLY one term:
#   A  output-KD only          (ZENSR_FKD_W=0)
#   B  output-KD + affinity KD (ZENSR_FKD_W=$W)
# Both use the same online teacher for the output target, the same data, the
# same seeds (fixed in the trainer), the same step budget, and the same box —
# sequentially, so neither arm is contending for the GPU with the other.
#
# The weight is not a guess: run a short probe with any non-zero ZENSR_FKD_W
# and read the `base` / `aff` components the trainer prints, then set W so the
# affinity term is a stated fraction of the total. An arm whose affinity term
# dwarfs reconstruction is not a KD arm, it is a different objective.
#
# Usage: run_fkd_pair.sh <weight> [steps] [batch] [lr]
# Env:   ZENSR_FKD_DATA (default ~/zensr-fkd), ZENSR_FKD_TCKPT (teacher .pth)
set -euo pipefail

W="${1:?affinity weight — calibrate it from a probe first, do not guess}"
STEPS="${2:-25000}"
BATCH="${3:-48}"
LR="${4:-3e-4}"
DATA="${ZENSR_FKD_DATA:-$HOME/zensr-fkd}"
TCKPT="${ZENSR_FKD_TCKPT:-$DATA/dejpeg11_teacher_100000.pth}"
PY="${ZENSR_PY:-$HOME/zensr-env/bin/python}"

[ -f "$TCKPT" ] || { echo "teacher checkpoint not found: $TCKPT" >&2; exit 2; }
cd "${ZENSR_REPO:-$HOME/zensr-ablation}"

run_arm() {
  local name="$1"
  local w="$2"
  # separate statements: under `set -u`, referencing a name assigned
  # earlier in the SAME `local` is an unbound-variable error
  local log="$HOME/fkd_${name}.log"
  echo "== arm $name (ZENSR_FKD_W=$w) -> $log"
  ZENSR_COMPILE="${ZENSR_COMPILE:-0}" \
  ZENSR_DATA="$DATA" ZENSR_SCALE=1 ZENSR_NF=24 ZENSR_NC=8 ZENSR_QBOOST=3 \
  ZENSR_FKD_TEACHER="$TCKPT" ZENSR_FKD_TNF=64 ZENSR_FKD_TNC=16 \
  ZENSR_FKD_W="$w" ZENSR_OUT="fkd_${name}" ZENSR_CKPT_EVERY=2000 \
    nice -n 5 "$PY" tools/train_people.py "$STEPS" "$BATCH" "$LR" > "$log" 2>&1
  echo "== arm $name done: $(tail -1 "$log")"
}

# Control first: if it fails, the comparison is void and no GPU time is spent
# on the treatment arm.
run_arm a_outkd 0
run_arm b_affinity "$W"
echo "PAIR_DONE" > "$HOME/fkd_pair.done"
