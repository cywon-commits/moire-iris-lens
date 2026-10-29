import sys,json,os,numpy as np
from pair3 import run
tag=sys.argv[1]; alpha=float(sys.argv[2]); V0=float(sys.argv[3]); defs=eval(sys.argv[4]); seed=sys.argv[5] if len(sys.argv)>5 else None
if os.path.exists(tag+'.json'): sys.exit()
x0=None
if seed:
    D=np.load(seed+'.npz'); x0=np.concatenate([D['ux'].ravel(),D['uy'].ravel()])
R=run(defs,free=True,V0=V0,alpha=alpha,lam=-0.5,x0=x0)
np.savez(tag+'.npz',x=R['x'],ux=R['ux'],uy=R['uy'],core=R['core']); json.dump(dict(E=R['E'],nit=int(R['nit'])),open(tag+'.json','w')); print(tag,R['E'],R['nit'],flush=True)
