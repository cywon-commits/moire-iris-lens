"""Relax a state in a large clamped box (Lp=148.5, same h=0.45), optionally seeded by embedding a small-box (Lp=72) state.
usage: bigrun.py TAG ALPHA V0 "DEFECTS" [SEED_NPZ_PATH_WITHOUT_EXT]"""
import sys,os,json,time,numpy as np
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','pair'))
from pair2 import run, c1
LP=148.5; h=0.45
tag,alpha,V0=sys.argv[1],float(sys.argv[2]),float(sys.argv[3]); defs=eval(sys.argv[4]); seed=sys.argv[5] if len(sys.argv)>5 else None
if os.path.exists(tag+'.json'): sys.exit()
N=int(round(2*LP/h))+1; x=np.linspace(-LP,LP,N); X,Y=np.meshgrid(x,x,indexing='ij')
k=0.0333333333333333*np.exp(1j*np.radians(alpha)); ubx=c1[0]+(k*(X+1j*Y)).real; uby=c1[1]+(k*(X+1j*Y)).imag
x0=None
if seed:
    S=np.load(seed+'.npz'); xs=S['x']; n=len(xs); off=int(round((xs[0]-x[0])/h)); assert abs(x[off]-xs[0])<1e-9
    Xs,Ys=np.meshgrid(xs,xs,indexing='ij'); ks=k; dux=S['ux']-(c1[0]+(ks*(Xs+1j*Ys)).real); duy=S['uy']-(c1[1]+(ks*(Xs+1j*Ys)).imag)
    ux=ubx.copy(); uy=uby.copy(); ux[off:off+n,off:off+n]+=dux; uy[off:off+n,off:off+n]+=duy
    x0=np.concatenate([ux.ravel(),uy.ravel()])
t=time.time(); R=run(defs,V0=V0,alpha=alpha,lam=-0.5,Lp=LP,x0=x0)
np.savez(tag+'.npz',x=R['x'],ux=R['ux'],uy=R['uy'],core=R['core']); json.dump(dict(E=R['E'],nit=int(R['nit']),t=time.time()-t,seed=seed),open(tag+'.json','w'))
print(tag,R['E'],R['nit'],round(time.time()-t),'s',flush=True)
