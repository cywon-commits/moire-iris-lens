import sys,json,numpy as np
from pair4 import run
a=float(sys.argv[1]); eps=float(sys.argv[2]); V0=float(sys.argv[3]); Lp=float(sys.argv[4]); r0=float(sys.argv[5])
R=run([],V0=V0,alpha=a,lam=-0.5,Lp=Lp,incl=(0.,0.,r0,eps),maxit=60000)
tag=f'inc_a{a:.0f}_e{eps:+.2f}_V{V0}_L{Lp:.0f}_r{r0:.0f}'
np.savez(tag+'.npz',x=R['x'],ux=R['ux'],uy=R['uy'],core=R['core'])
print(tag,R['E'],R['nit'],round(R['t']),flush=True)
