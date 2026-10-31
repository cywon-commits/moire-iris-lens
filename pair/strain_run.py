# homogeneous moire strain: U = Ua z + Ub conj(z); registry ub = c1 + kap(z - U)
import sys,json,numpy as np
from pair4 import run
alpha=float(sys.argv[1]); mode=sys.argv[2]; amp=float(sys.argv[3]); Lp=float(sys.argv[4]) if len(sys.argv)>4 else 72.
Ua,Ub={'d':(amp,0),'r':(1j*amp,0),'s1':(0,amp),'s2':(0,1j*amp),'0':(0,0)}[mode]
R=run([],V0=0.005,alpha=alpha,lam=-0.5,Lp=Lp,Ua=Ua,Ub=Ub,maxit=40000)
# split energies
EG=R['EG']; 
tag=f'st_a{alpha:.0f}_{mode}_{amp:+.3f}_L{Lp:.0f}'
np.savez(tag+'.npz',x=R['x'],ux=R['ux'],uy=R['uy'])
json.dump(dict(alpha=alpha,mode=mode,amp=amp,Lp=Lp,E=R['E'],nit=int(R['nit'])),open(tag+'.json','w'))
print(tag,R['E'],R['nit'],round(R['t']),flush=True)
