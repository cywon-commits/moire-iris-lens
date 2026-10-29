"""Hyperplane-constrained relaxation between two minima (drag), optionally with local tangents (plane string)."""
import sys,json,numpy as np
from scipy.optimize import minimize
from pair import run
def setup(lam,d):
    EG,N,x=run([(-d/2,0.,4.),(d/2,0.,4.)],V0=0.005,alpha=90.,lam=lam,want_EG=True); return EG,N,x
def load(f):
    D=np.load(f+'.npz'); return np.concatenate([D['ux'].ravel(),D['uy'].ravel()]),D['core']
def relax_plane(EG,z0,t,maxit=20000):
    t=t/np.linalg.norm(t)
    def f(z):
        E,G=EG(z); G=G-np.dot(G,t)*t; return E,G
    r=minimize(f,z0,jac=True,method='L-BFGS-B',options=dict(maxiter=maxit,maxfun=2*maxit,gtol=1e-9,ftol=1e-14,maxcor=30))
    return r.x,r.fun
if __name__=='__main__':
    lam=float(sys.argv[1]); d=float(sys.argv[2]); A,B,pre=sys.argv[3],sys.argv[4],sys.argv[5]; xis=[float(v) for v in sys.argv[6].split(',')]
    EG,N,x=setup(lam,d); zA,core=load(A); zB,_=load(B)
    mask=np.ones(2*N*N); fix=np.zeros((N,N),bool); fix[:2]=fix[-2:]=True; fix[:,:2]=fix[:,-2:]=True
    bad=(fix|core).ravel(); mask[np.concatenate([bad,bad])]=0
    t=(zB-zA)*mask
    for xi in xis:
        z,E=relax_plane(EG,zA+xi*(zB-zA),t)
        np.save(f'{pre}_xi{xi:.2f}.npy',z); print(pre,xi,E,flush=True)
