import sys,json,numpy as np
from pair import run
lam=float(sys.argv[1]); alpha=float(sys.argv[2]); prefix=sys.argv[3]
for src in sys.argv[4:]:
    d=float(src.split('_d')[1]); D=np.load(src+'.npz'); x0=np.concatenate([D['ux'].ravel(),D['uy'].ravel()])
    R=run([(-d/2,0.,4.),(d/2,0.,4.)],V0=0.005,alpha=alpha,lam=lam,x0=x0)
    t=f'{prefix}_d{d}'; np.savez(t+'.npz',x=R['x'],ux=R['ux'],uy=R['uy'],core=R['core']); json.dump(dict(d=d,E=R['E']),open(t+'.json','w')); print(t,R['E'],flush=True)
