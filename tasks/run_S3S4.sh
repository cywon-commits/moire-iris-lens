#!/bin/bash
# wait for the S2 scanner, then run S3 and S4 (resumable)
cd "$(dirname "$0")/../pair"
while pgrep -f "pscan_scan.py ../tasks/jobs_screen_S2.txt" >/dev/null; do sleep 60; done
python3 pscan_scan.py ../tasks/jobs_screen_S3.txt 4 screen_run.py > scan_screen_S3.log 2>&1
python3 pscan_scan.py ../tasks/jobs_screen_S4.txt 4 screen_run.py > scan_screen_S4.log 2>&1
