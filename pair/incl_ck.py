"""Checkpointed moire eigen-inclusion run (resumes after interruption from <tag>_ck.npz).
usage: python incl_ck.py ALPHA EPS V0 LP R0 [CHUNK]
  ALPHA 90=twist, 0=mismatch ; EPS moire eigen-dilation inside disk |z|<R0 ; V0=0 gives the bare-elastic reference.
Energy E is stored in the npz."""
import sys,os,numpy as np
from pair4 import run
a=float(sys.argv[1]); eps=float(sys.argv[2]); V0=float(sys.argv[3]); Lp=float(sys.argv[4]); r0=float(sys.argv[5])
chunk=int(sys.argv[6]) if len(sys.argv)>6 else 400
LAM=float(os.environ.get('INC_LAM','-0.5'))
tag=f'inc_a{a:.0f}_e{eps:+.3f}_V{V0}_L{Lp:.0f}_r{r0:.0f}'+('' if LAM==-0.5 else f'_lam{LAM}')
if os.path.exists(tag+'.npz'): print(tag,'done already'); sys.exit()
x0=None; tot=0; Eold=None
if os.path.exists(tag+'_ck.npz'):
    C=np.load(tag+'_ck.npz'); x0=np.concatenate([C['ux'].ravel(),C['uy'].ravel()]); tot=int(C['tot']); Eold=float(C['E'])
while True:
    R=run([],V0=V0,alpha=a,lam=LAM,Lp=Lp,incl=(0.,0.,r0,eps),maxit=chunk,x0=x0)
    tot+=R['nit']; x0=np.concatenate([R['ux'].ravel(),R['uy'].ravel()])
    np.savez(tag+'_ck.npz',x=R['x'],ux=R['ux'],uy=R['uy'],core=R['core'],tot=tot,E=R['E'])
    print(tag,'chunk',R['nit'],'tot',tot,'E',R['E'],flush=True)
    # stop: optimizer converged inside the chunk, or energy change per chunk < 1e-6 (L-BFGS restarts each chunk)
    if R['nit']<chunk or (Eold is not None and abs(Eold-R['E'])<1e-6): break
    Eold=R['E']
os.replace(tag+'_ck.npz',tag+'.npz'); print(tag,'DONE',R['E'],tot,flush=True)
