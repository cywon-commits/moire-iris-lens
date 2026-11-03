"""S1-S4 (tasks/TASK_screening_scan.md): one checkpointed relaxation for the screening scan.
usage: python screen_run.py ALPHA CABS PHIM PLACE V0 LAM INIT LP      (defect run; INIT a = linear lens field, b = relaxed background)
       python screen_run.py bg ALPHA V0 LAM LP                        (defect-free background on the same grid)
r_h = 2.4 sqrt(|C|/4) (as P1); stop when |dE| per 400-iteration chunk < 1e-5 (pscan_run.relax).
V0 = 0 is the pure elastic reference: its background is the analytic u_b (written directly, no relaxation)."""
import sys,os,numpy as np
from pscan_run import relax,place,rh_of,kap
from pair4 import run
s3=np.sqrt(3.); c1=np.array([0.5,1/(2*s3)])
def btag(alpha,V0,lam,Lp): return f'scb_a{alpha:.0f}_V{V0:g}_l{lam:g}_L{Lp:.0f}'
def dtag(alpha,cabs,phim,pl,V0,lam,init,Lp): return f'scd_a{alpha:.0f}_C{cabs:g}_p{phim:+.0f}_{pl}_V{V0:g}_l{lam:g}_{init}_L{Lp:.0f}'
def background(alpha,V0,lam,Lp):
    t=btag(alpha,V0,lam,Lp)
    if V0>0: relax(t,dict(defects=[],V0=V0,alpha=alpha,lam=lam,Lp=Lp)); return t
    if not os.path.exists(t+'.npz'):
        EG,N,x=run([],V0=0.,alpha=alpha,lam=lam,Lp=Lp,want_EG=True)
        X,Y=np.meshgrid(x,x,indexing='ij'); W=kap(alpha)*(X+1j*Y); ux=c1[0]+W.real; uy=c1[1]+W.imag
        E=float(EG(np.concatenate([ux.ravel(),uy.ravel()]))[0])
        np.savez(t+'.npz',x=x,ux=ux,uy=uy,core=np.zeros_like(ux,bool),tot=0,E=E); print(t,'analytic u_b, E',E,flush=True)
    return t
if __name__=='__main__':
    if sys.argv[1]=='bg':
        background(float(sys.argv[2]),float(sys.argv[3]),float(sys.argv[4]),float(sys.argv[5])); sys.exit()
    alpha,cabs,phim=float(sys.argv[1]),float(sys.argv[2]),float(sys.argv[3]); pl=sys.argv[4]
    V0,lam=float(sys.argv[5]),float(sys.argv[6]); init=sys.argv[7]; Lp=float(sys.argv[8])
    z0=place(pl,alpha); C=cabs*np.exp(1j*np.radians(alpha+phim)); x0=None
    if init=='b':
        import time
        b=btag(alpha,V0,lam,Lp)
        if V0==0: background(alpha,V0,lam,Lp)
        while not os.path.exists(b+'.npz'): time.sleep(60)
        B=np.load(b+'.npz'); x0=np.concatenate([B['ux'].ravel(),B['uy'].ravel()])
    relax(dtag(alpha,cabs,phim,pl,V0,lam,init,Lp),dict(defects=[(z0.real,z0.imag,C)],V0=V0,alpha=alpha,lam=lam,Lp=Lp,rh=rh_of(cabs,'s')),x0=x0)
