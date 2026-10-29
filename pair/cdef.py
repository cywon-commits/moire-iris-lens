import sys,json,numpy as np
from pair2 import run
alpha=float(sys.argv[1]); C=complex(sys.argv[2]); tag=sys.argv[3]; d=16.
R=run([(d/2,0.,C)],V0=0.005,alpha=alpha,lam=-0.5)
np.savez(tag+'.npz',x=R['x'],ux=R['ux'],uy=R['uy'],core=R['core']); json.dump(dict(alpha=alpha,C=str(C),E=R['E']),open(tag+'.json','w')); print(tag,R['E'],flush=True)
