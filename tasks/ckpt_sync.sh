#!/bin/bash
# Snapshot pair/inc_*.npz (checkpoints + finished runs) and run logs to a single-commit branch on origin (force-push).
# usage: tasks/ckpt_sync.sh            (run from anywhere inside the repo)
set -e
cd "$(git rev-parse --show-toplevel)"
BR=${CKPT_BRANCH:-claude/github-upload-d5q58c-ckpt}
T=$(mktemp -d); I=$(mktemp -d); trap 'rm -rf "$T" "$I"' EXIT
mkdir -p "$T/pair"
for f in pair/inc_*.npz pair/log_inc_*.txt pair/scan_*.log; do [ -e "$f" ] && cp "$f" "$T/pair/"; done
# drop snapshots caught mid-write (np.savez is not atomic)
python3 -I - "$T/pair" <<'PY'
import sys,glob,os,numpy as np
for f in glob.glob(sys.argv[1]+'/*.npz'):
    try: d=np.load(f); [d[k] for k in d.files]
    except Exception as e: print('skip (partial)',os.path.basename(f)); os.remove(f)
PY
export GIT_INDEX_FILE="$I/index"
git --work-tree="$T" add -f -A .
tree=$(git write-tree)
c=$(git commit-tree "$tree" -m "inclusion-run checkpoints $(date -u +%FT%TZ)")
for i in 1 2 3 4 5; do git push -q -f origin "$c:refs/heads/$BR" && break; sleep $((2**i)); done
echo "synced $(ls "$T/pair" | wc -l) files -> $BR at $(date -u +%T)"
