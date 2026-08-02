#!/usr/bin/env bash
# Pull intermediate training checkpoints off a remote box, repeatedly.
#
# Why this exists: on 2026-08-01 a student run on a dual-boot box reached step
# 22.5k of 200k and then the box rebooted into its other OS. The launcher only
# collected the FINAL checkpoint, so 22.5k steps of progress became unreachable
# until that box next boots back. Any run on a box we do not exclusively own —
# and any run long enough to outlive a reboot — needs its checkpoints copied
# off as they are written, not at the end.
#
# Boxes come from the environment so no host/LAN details live in the repo:
#   ZENSR_BOXES="name=user@host name=user@host ..."
#
# Usage: pull_ckpts.sh <box-name> [remote-dir] [local-dir] [interval-seconds]
#   pull_ckpts.sh jason                      # defaults, loops every 10 min
#   pull_ckpts.sh lianli ~/zensr-big ~/tmp/zensr-ckpts/lianli 300
#   ONESHOT=1 pull_ckpts.sh jason            # single pass, for cron
set -euo pipefail

NAME="${1:?box name (key in ZENSR_BOXES)}"
REMOTE_DIR="${2:-\~/zensr-train}"
LOCAL_DIR="${3:-$HOME/tmp/zensr-ckpts/$NAME}"
INTERVAL="${4:-600}"

declare -A BOX
for kv in ${ZENSR_BOXES:-}; do BOX[${kv%%=*}]="${kv#*=}"; done
HOST="${BOX[$NAME]:-}"
if [ -z "$HOST" ]; then
  echo "set ZENSR_BOXES, e.g. ZENSR_BOXES='$NAME=user@host'" >&2
  exit 2
fi

mkdir -p "$LOCAL_DIR"
echo "pulling $NAME ($HOST):$REMOTE_DIR/*.pth -> $LOCAL_DIR every ${INTERVAL}s"

while :; do
  # --ignore-existing: checkpoints are immutable once written, so never re-copy.
  # A failed pass is expected (box asleep, rebooted, mid-write) and must not
  # kill the loop — that would defeat the purpose.
  if rsync -a --ignore-existing --timeout=120 \
       --include='*_[0-9]*.pth' --include='*.json' --include='*.log' \
       --exclude='*' \
       "$HOST":"$REMOTE_DIR"/ "$LOCAL_DIR"/ 2>/dev/null; then
    n=$(find "$LOCAL_DIR" -name '*.pth' | wc -l)
    newest=$(ls -t "$LOCAL_DIR"/*.pth 2>/dev/null | head -1)
    printf '%s  %s: %d ckpt(s) held%s\n' \
      "$(date -u +%H:%M:%SZ)" "$NAME" "$n" "${newest:+, newest $(basename "$newest")}"
  else
    printf '%s  %s: unreachable (asleep, rebooted, or mid-write) — retrying\n' \
      "$(date -u +%H:%M:%SZ)" "$NAME"
  fi
  [ -n "${ONESHOT:-}" ] && break
  sleep "$INTERVAL"
done
