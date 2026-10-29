python3 pair.py none 0 0.005 0 x
for d in 8 26; do python3 pair.py pair $d 0.005 0 x; python3 pair.py R $d 0.005 0 x; done
for d in 12 40; do python3 pair.py pair $d 0.005 90 y; python3 pair.py R $d 0.005 90 y; done
