# checkpointed single large defect: python big_ck.py ALPHA CRE CIM LP RH TAG
import sys,os,numpy as np
from pair4 import run
a=float(sys.argv[1]); C=complex(float(sys.argv[2]),float(sys.argv[3])); Lp=float(sys.argv[4]); rh=float(sys.argv[5]); tag=sys.argv[6]
if os.path.exists(tag+'.npz'): print('done'); sys.exit()
x0=None; tot=0; Eold=None
if os.path.exists(tag+'_ck.npz'):
    D=np.load(tag+'_ck.npz'); x0=np.concatenate([D['ux'].ravel(),D['uy'].ravel()]); tot=int(D['tot']); Eold=float(D['E'])
while True:
    R=run([(0.,0.,C)],V0=0.005,alpha=a,lam=-0.5,Lp=Lp,rh=rh,maxit=400,x0=x0)
    tot+=R['nit']; x0=np.concatenate([R['ux'].ravel(),R['uy'].ravel()])
    np.savez(tag+'_ck.npz',x=R['x'],ux=R['ux'],uy=R['uy'],core=R['core'],tot=tot,E=R['E'])
    print(tag,'tot',tot,'E',R['E'],flush=True)
    if R['nit']<400 or (Eold is not None and abs(Eold-R['E'])<1e-5): break
    Eold=R['E']
os.replace(tag+'_ck.npz',tag+'.npz'); print(tag,'DONE',flush=True)
