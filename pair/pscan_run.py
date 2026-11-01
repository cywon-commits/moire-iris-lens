"""Partition scan (tasks/TASK_partition_scan.md, P1): one checkpointed relaxation.
usage: python pscan_run.py ALPHA CABS PHIM PLACE INIT [RHMODE]
  ALPHA 90=twist, 0=mismatch ; CABS=|C| ; PHIM=phi_m in degrees (arg C = ALPHA+PHIM)
  PLACE in AA, AB, SP, GEN (defect centre z0, see place()) ; INIT a=linear lens field, b=relaxed background
  RHMODE s (default, r_h=2.4*sqrt(|C|/4): fixed core strain) or f (r_h=2.4)
  ALPHA=bg: relax the defect-free background for box CABS (=Lp) -> pbg_a{PHIM}_L{Lp}.npz (PHIM is then alpha)
Resumes from <tag>_ck.npz; stops when |dE| per 400-iteration chunk < 1e-5 (as big_ck.py)."""
import sys,os,numpy as np
from pair4 import run
s3=np.sqrt(3.); th=1/30.; LM=30.
a1=np.array([1.,0.]); a2=np.array([.5,s3/2])
def kap(alpha): return th*np.exp(1j*np.radians(alpha))
def place(name,alpha):
    k=kap(alpha)
    if name=='AA': d=0j
    elif name=='AB': v=(a1+a2)/3; d=complex(*v)            # c_AB-c_AA: up triangle (AB), see topo.labels_on
    elif name=='SP': v=-a1/2; d=complex(*v)                # wall midpoint between AB and BA (saddle of V)
    elif name=='GEN': return 0.37*LM*np.exp(1j*np.radians(17.))
    else: raise ValueError(name)
    return d/k
def box(cabs): return 72. if cabs<=8 else (108. if cabs<=16 else 150.)
def rh_of(cabs,mode): return 2.4*np.sqrt(cabs/4.) if mode=='s' else 2.4
def tag(alpha,cabs,phim,pl,init,mode='s'):
    return f'ps_a{alpha:.0f}_C{cabs:g}_p{phim:+.0f}_{pl}_{init}'+('' if mode=='s' else '_rhf')
def bgtag(alpha,Lp): return f'pbg_a{alpha:.0f}_L{Lp:.0f}'
def relax(t,kw,x0=None,chunk=400,tol=1e-5):
    if os.path.exists(t+'.npz'): print(t,'done already'); return
    tot=0; Eold=None
    if os.path.exists(t+'_ck.npz'):
        C=np.load(t+'_ck.npz'); x0=np.concatenate([C['ux'].ravel(),C['uy'].ravel()]); tot=int(C['tot']); Eold=float(C['E'])
    while True:
        R=run(maxit=chunk,x0=x0,**kw); tot+=R['nit']; x0=np.concatenate([R['ux'].ravel(),R['uy'].ravel()])
        np.savez(t+'_ck.npz',x=R['x'],ux=R['ux'],uy=R['uy'],core=R['core'],tot=tot,E=R['E'])
        print(t,'chunk',R['nit'],'tot',tot,'E',R['E'],flush=True)
        if R['nit']<chunk or (Eold is not None and abs(Eold-R['E'])<tol): break
        Eold=R['E']
    os.replace(t+'_ck.npz',t+'.npz'); print(t,'DONE',R['E'],tot,flush=True)
if __name__=='__main__':
    if sys.argv[1]=='bg':
        Lp=float(sys.argv[2]); alpha=float(sys.argv[3])
        relax(bgtag(alpha,Lp),dict(defects=[],V0=0.005,alpha=alpha,lam=-0.5,Lp=Lp))
        sys.exit()
    alpha=float(sys.argv[1]); cabs=float(sys.argv[2]); phim=float(sys.argv[3]); pl=sys.argv[4]; init=sys.argv[5]
    mode=sys.argv[6] if len(sys.argv)>6 else 's'
    Lp=box(cabs); rh=rh_of(cabs,mode); z0=place(pl,alpha); C=cabs*np.exp(1j*np.radians(alpha+phim))
    x0=None
    if init=='b':
        import time
        while not os.path.exists(bgtag(alpha,Lp)+'.npz'): time.sleep(60)   # background still relaxing
        B=np.load(bgtag(alpha,Lp)+'.npz'); x0=np.concatenate([B['ux'].ravel(),B['uy'].ravel()])
    relax(tag(alpha,cabs,phim,pl,init,mode),dict(defects=[(z0.real,z0.imag,C)],V0=0.005,alpha=alpha,lam=-0.5,Lp=Lp,rh=rh),x0=x0)
