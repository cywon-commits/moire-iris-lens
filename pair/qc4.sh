while pgrep -f "bash qc2.sh" >/dev/null; do sleep 20; done
python3 blind.py
