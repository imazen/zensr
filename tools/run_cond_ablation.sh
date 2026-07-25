#!/usr/bin/env bash
# Distributed conditioning ablation (S2a/S7/S10) over the household LAN fleet.
# Local box generates data + orchestrates; each box trains one matched arm:
#   lianli (RTX 2080)  -> C0 none    jason (RTX 3070) -> C1 scalar
#   ian    (RTX 3070?) -> C2 dmap
# Usage: run_cond_ablation.sh sync|launch|status|collect
set -euo pipefail
DATA=~/tmp/zensr-dejpeg-v4
declare -A BOX=( [lianli]="lilith@192.168.50.27" [jason]="zen@192.168.50.148" [ian]="zen@192.168.50.193" )
declare -A ARM=( [lianli]="none" [jason]="scalar" [ian]="dmap" )
STEPS=${STEPS:-14000}

case "${1:?sync|launch|status|collect}" in
  sync)
    for b in "${!BOX[@]}"; do
      echo "== sync $b"
      rsync -a --info=progress2 "$DATA"/{lr_u8.npy,hr_u8.npy,cond_scalar_f32.npy,dmap_u16.npy,pairs.tsv,meta.json} \
        ~/work/zen/zensr/tools/train_cond.py "${BOX[$b]}":~/zensr-ablation/ &
    done; wait ;;
  launch)
    for b in "${!BOX[@]}"; do
      a="${ARM[$b]}"
      echo "== launch $b arm=$a"
      ssh -o BatchMode=yes "${BOX[$b]}" "cd ~/zensr-ablation && \
        ZENSR_DATA=~/zensr-ablation ZENSR_COND=$a ZENSR_OUT=dejpeg5_$a \
        ZENSR_INIT=~/zensr-ablation/dejpeg4_policy_10000.pth ZENSR_QBOOST=3 \
        nohup ~/zensr-env/bin/python train_cond.py $STEPS 64 7e-5 > train_$a.log 2>&1 & echo started"
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
