# usage: python e1place.py BASELINE_NPZ_STEM ALPHA LP R0 [R0 ...]  -> E1 over 16 ring-centre placements in one moire cell
import numpy as np,json,sys
from e1ring import E1
base=sys.argv[1] if len(sys.argv)>1 else 'inc_a90_e+0.00_V0.005_L150_r60_ck'; al=float(sys.argv[2]) if len(sys.argv)>2 else 90.; Lp=float(sys.argv[3]) if len(sys.argv)>3 else 150.; R0s=[float(v) for v in sys.argv[4:]] or [45.,70.]; res={}
s3=np.sqrt(3)
# moire cell (twist): lattice vectors 30*(-i) and 30*(-i)*e^{i pi/3}; sample 4x4 fractional positions
k=np.exp(1j*np.radians(al)); T1=30/k; T2=30*np.exp(1j*np.pi/3)/k
for r0 in R0s:
    v=[]
    for a in np.arange(4)/4:
        for b in np.arange(4)/4:
            z0=a*T1+b*T2; v.append(E1(base,al,Lp,r0,z0.real,z0.imag))
    v=np.array(v); res[r0]=v.tolist(); print(r0,'mean',v.mean(),'std',v.std(),'min',v.min(),'max',v.max(),flush=True)
json.dump(res,open(f'e1place_a{al:.0f}_L{Lp:.0f}.json','w'))
