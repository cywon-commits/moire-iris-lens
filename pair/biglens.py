import sys,numpy as np,json
from pair2 import run
a=float(sys.argv[1]); C=complex(sys.argv[2]); rh=float(sys.argv[3]); Lp=float(sys.argv[4])
R=run([(0.,0.,C)],V0=0.005,alpha=a,lam=-0.5,Lp=Lp,rh=rh,maxit=30000)
tag=f'biglens_a{a:.0f}_C{C.real:.0f}{C.imag:+.0f}j_r{rh:.0f}_L{Lp:.0f}'
np.savez(tag+'.npz',x=R['x'],ux=R['ux'],uy=R['uy'],core=R['core']); print(tag,R['E'],R['nit'],round(R['t']),flush=True)
