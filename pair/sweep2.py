import sys,json,numpy as np
from pair import run
direc,alpha,lam,start,prefix=sys.argv[1],float(sys.argv[2]),float(sys.argv[3]),sys.argv[4],sys.argv[5]
ds=[float(v) for v in sys.argv[6].split(',')]
src=start
for d in ds:
    D=np.load(src+'.npz'); x0=np.concatenate([D['ux'].ravel(),D['uy'].ravel()])
    R=run([(-d/2,0.,4.),(d/2,0.,4.)],V0=0.005,alpha=alpha,lam=lam,x0=x0)
    src=f'{prefix}{direc}_d{d}'
    np.savez(src+'.npz',x=R['x'],ux=R['ux'],uy=R['uy'],core=R['core']); json.dump(dict(d=d,E=R['E']),open(src+'.json','w'))
    print(src,R['E'],flush=True)
