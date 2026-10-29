import sys,json,numpy as np
from pair import run, c1
for src in sys.argv[1:]:
    d=float(src.split('_d')[1]); D=np.load(src+'.npz'); ux,uy=D['ux'],D['uy']
    uxs=0.5*(ux+ux[:,::-1]); uys=0.5*(uy-uy[:,::-1])+c1[1]
    R=run([(-d/2,0.,4.),(d/2,0.,4.)],V0=0.005,alpha=0.,lam=-0.5,mirror=True,x0=np.concatenate([uxs.ravel(),uys.ravel()]))
    t=f'{src}_symz'; json.dump(dict(d=d,E=R['E']),open(t+'.json','w')); print(t,R['E'],flush=True)
