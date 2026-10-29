for d in 12 20 32 50; do python3 pair.py pair $d 0.005; python3 pair.py L $d 0.005; python3 pair.py R $d 0.005; python3 pair.py pair $d 0; python3 pair.py R $d 0; done
python3 pair.py R 0 0.005
python3 pair.py none 0 0.005
