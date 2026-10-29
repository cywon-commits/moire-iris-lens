for dg in 0.60 0.85 1.20; do python3 wrk.py "N_$dg" "H0=0.,rho=4.,xc=0.,yc=6.,d1=0.3,dg=$dg,kap=0.01"; done
