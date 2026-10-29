import numpy as np, sys
from hexcore import *
V3=float(sys.argv[1])
R=run_periodic([0,1j/16],rigid0=True,V0=0.05,V3=V3)
U=R['U']; dx=U[1][0]-U[0][0]; dy=U[1][1]-U[0][1]
np.savez(f't1_V3_{V3}.npz',x=R['x'],y=R['y'],dx=dx,dy=dy,E=R['E'])
print(V3,R['E'],R['nit'],R['msg'],round(R['t']),'s',flush=True)
