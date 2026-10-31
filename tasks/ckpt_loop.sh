#!/bin/bash
# Sync checkpoints every ${CKPT_EVERY:-1200}s while any incl_ck.py runs; one final sync after they stop.
cd "$(dirname "$0")"
while pgrep -f incl_ck.py >/dev/null; do ./ckpt_sync.sh || echo "sync failed $(date -u +%T)"; sleep ${CKPT_EVERY:-1200}; done
./ckpt_sync.sh
