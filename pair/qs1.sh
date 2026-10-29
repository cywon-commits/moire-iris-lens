export PAIR_LAM=-0.5
for a in "none 0 0.005" "pair 8 0.005" "R 8 0.005" "pair 26 0.005" "R 26 0.005" "pair 56 0.005"; do
  set -- $a; t="$1_d$(python3 -c "print(float($2))")_V$3_a0_x_sym_lam-0.5"; [ -f $t.json ] || python3 pair.py $1 $2 $3 0 x sym; done
