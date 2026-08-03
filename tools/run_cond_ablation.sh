#!/usr/bin/env bash
# Distributed conditioning ablation (S2a/S7/S10) over the household LAN fleet.
# Local box generates data + orchestrates; each box trains one matched arm:
#   lianli (RTX 2080)  -> C0 none    node-2 (RTX 3070) -> C1 scalar
#   node-3    (RTX 3070?) -> C2 dmap
# Usage: run_cond_ablation.sh sync|launch|status|collect
set -euo pipefail
DATA=~/tmp/zensr-dejpeg-v4
# Boxes come from the environment so no host/LAN details live in the repo:
#   ZENSR_BOXES="name=user@host name=user@host ..."
declare -A BOX
for kv in ${ZENSR_BOXES:-}; do BOX[${kv%%=*}]="${kv#*=}"; done
if [ ${#BOX[@]} -eq 0 ]; then
  echo "set ZENSR_BOXES, e.g. ZENSR_BOXES='a=user@host-a b=user@host-b'" >&2; exit 2
fi
declare -A ARM=( [lianli]="none" [node-2]="scalar" [node-3]="dmap" )
STEPS=${STEPS:-14000}
BATCH=${BATCH:-48}

case "${1:?sync|launch|status|collect}" in
  sync)
    for b in "${!BOX[@]}"; do
      echo "== sync $b"
      rsync -a --info=progress2 "$DATA"/{lr_u8.npy,hr_u8.npy,cond_scalar_f32.npy,dmap_u16.npy,pairs.tsv,meta.json} \
        ~/work/zen/zensr/tools/train_cond.py ~/tmp/zensr-dejpeg-v3/policy/dejpeg4_policy_10000.pth "${BOX[$b]}":~/zensr-ablation/ &
    done; wait ;;
  launch)
    for b in "${!BOX[@]}"; do
      a="${ARM[$b]}"
      echo "== launch $b arm=$a"
      ssh -o BatchMode=yes "${BOX[$b]}" "cd ~/zensr-ablation && \
        ZENSR_DATA=~/zensr-ablation ZENSR_COND=$a ZENSR_OUT=dejpeg5_$a \
        ZENSR_INIT=~/zensr-ablation/dejpeg4_policy_10000.pth ZENSR_QBOOST=3 \
        nohup ~/zensr-env/bin/python train_cond.py $STEPS $BATCH 7e-5 > train_$a.log 2>&1 & echo started"
    done ;;
  status)
    for b in "${!BOX[@]}"; do
      printf "[%s/%s] " "$b" "${ARM[$b]}"
      ssh -o BatchMode=yes -o ConnectTimeout=6 "${BOX[$b]}" \
        "tail -1 ~/zensr-ablation/train_${ARM[$b]}.log 2>/dev/null || echo no-log" 2>/dev/null || echo unreachable
    done ;;
  collect)
    mkdir -p "$DATA"/results
    for b in "${!BOX[@]}"; do
      a="${ARM[$b]}"
      scp -q "${BOX[$b]}":~/zensr-ablation/dejpeg5_${a}_final.pth "$DATA"/results/ 2>/dev/null && echo "got $a" || echo "missing $a"
      ssh -o BatchMode=yes "${BOX[$b]}" "grep -E 'STRATA|step $STEPS' ~/zensr-ablation/train_${a}.log" 2>/dev/null | sed "s/^/[$a] /"
    done ;;
esac
