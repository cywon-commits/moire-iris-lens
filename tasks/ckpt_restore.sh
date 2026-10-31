#!/bin/bash
# Restore checkpoints from the ckpt branch into pair/ (fresh container), then resume with incl_scan.py.
set -e
cd "$(git rev-parse --show-toplevel)"
BR=${CKPT_BRANCH:-claude/github-upload-d5q58c-ckpt}
git fetch -q origin "$BR"
git archive FETCH_HEAD | tar -x -C .
ls pair/inc_*.npz
