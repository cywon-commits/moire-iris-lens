import sys,json,numpy as np
from pair import run
d=float(sys.argv[1]); src=sys.argv[2]; C=4.0
D=np.load(src+'.npz'); x0=np.concatenate([D['ux'].ravel(),D['uy'].ravel()])
R=run([(-d/2,0.,C),(d/2,0.,C)],V0=0.005,alpha=90.,lam=-0.5,x0=x0)
tag=f'pair_d{d}_V0.005_lam-0.5_from_{src.split("_")[1]}'
np.savez(tag+'.npz',x=R['x'],ux=R['ux'],uy=R['uy'],core=R['core'])
json.dump(dict(d=d,E=R['E'],src=src),open(tag+'.json','w')); print(tag,R['E'],flush=True)
