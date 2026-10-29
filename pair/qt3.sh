export PAIR_LAM=-0.5
for a in $1; do for m in pair R; do t="${m}_d$(python3 -c "print(float($a))")_V0.005_lam-0.5"; [ -f $t.json ] || python3 pair.py $m $a 0.005 90 x; done; done
