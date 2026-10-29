for d in 16 40; do python3 pair.py pair $d 0.005 0 x; python3 pair.py R $d 0.005 0 x; done
python3 pair.py pair 26 0.005 90 y; python3 pair.py R 26 0.005 90 y
python3 pair.py L 26 0.005 0 x
