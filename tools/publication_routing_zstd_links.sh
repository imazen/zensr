#!/bin/sh
# Frozen auditor adapter: resolve archive links before the unchanged zstd read.
# Only the auditor's read-to-stdout invocation is accepted; no files are written.
set -eu
test "$#" -eq 4
test "$1" = -q && test "$2" = -d && test "$3" = -c
resolved=$(readlink -f -- "$4")
exec "${ZENSR_AUDIT_ZSTD_NATIVE:-/usr/bin/zstd}" "$1" "$2" "$3" "$resolved"
