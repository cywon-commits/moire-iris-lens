for d in 8 16 26 40; do python3 pair.py pair $d 0.005; python3 pair.py L $d 0.005; python3 pair.py R $d 0.005; python3 pair.py pair $d 0; python3 pair.py R $d 0; done
python3 pair.py merged 0 0.005
