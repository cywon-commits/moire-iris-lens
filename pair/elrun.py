import sys,json,numpy as np
from pair2 import run
tag=sys.argv[1]; defs=eval(sys.argv[2])
R=run(defs,V0=0.0,alpha=0.,lam=-0.5)
np.savez(tag+'.npz',x=R['x'],ux=R['ux'],uy=R['uy'],core=R['core']); print(tag,R['E'],flush=True)
