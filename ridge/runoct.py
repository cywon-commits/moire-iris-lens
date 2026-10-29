import numpy as np, sys
from relax2d import run
Y,V0=3.,0.05; Lam=0.6; lb=0.5; P=np.sqrt(Lam*Y*V0)
for mode in ['n2','oct']:
    R=run(L=12.,V0=V0,kap=P*lb**2,Kz=P/lb**2,mode=mode,maxit=80000)
    np.savez(f'r2d_{mode}.npz',x=R['x'],ux=R['ux'],uy=R['uy'],h=R['h'])
    print(mode,R['nit'],R['msg'],round(R['t']),'s',flush=True)
