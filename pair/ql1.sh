for d in 8 26 56; do for a in "pair $d 0.005" "R $d 0.005" "pair $d 0" "R $d 0"; do bash runskip.sh -0.5 $a 0 x; done; done
