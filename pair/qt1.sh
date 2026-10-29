export PAIR_LAM=-0.5
for a in "none 0" "pair 16" "R 16" "pair 40" "R 40" "R 56"; do set -- $a
  t="$1_d$(python3 -c "print(float($2))")_V0.005_lam-0.5"; [ -f $t.json ] || python3 pair.py $1 $2 0.005 90 x; done
