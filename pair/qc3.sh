while pgrep -f "bash qc1.sh" >/dev/null; do sleep 20; done
for d in 46 56 66; do python3 pair.py R $d 0.005; done
