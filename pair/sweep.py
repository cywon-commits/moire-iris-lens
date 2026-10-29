import sys,json,numpy as np
from pair import run
direc=sys.argv[1]; ds=[8.,16.,21.,26.,30.,34.,40.,48.,56.]
if direc=='in': ds=ds[::-1]
src=f'pair_d{ds[0]}_V0.005_lam-0.5'
for d in ds[1:]:
    D=np.load(src+'.npz'); x0=np.concatenate([D['ux'].ravel(),D['uy'].ravel()])
    R=run([(-d/2,0.,4.),(d/2,0.,4.)],V0=0.005,alpha=90.,lam=-0.5,x0=x0)
    src=f'sw{direc}_d{d}'
    np.savez(src+'.npz',x=R['x'],ux=R['ux'],uy=R['uy'],core=R['core']); json.dump(dict(d=d,E=R['E']),open(src+'.json','w'))
    print(src,R['E'],flush=True)
