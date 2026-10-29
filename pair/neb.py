import sys,json,time,numpy as np
from pair import run
"""NEB with climbing image, FIRE optimizer. usage: neb.py lam d stateA stateB tag [M] [nsteps]"""
lam=float(sys.argv[1]); d=float(sys.argv[2]); A_,B_=sys.argv[3],sys.argv[4]; tag=sys.argv[5]
M=int(sys.argv[6]) if len(sys.argv)>6 else 11; NS=int(sys.argv[7]) if len(sys.argv)>7 else 4000
EG,N,x=run([(-d/2,0.,4.),(d/2,0.,4.)],V0=0.005,alpha=90.,lam=lam,want_EG=True)
def load(f): D=np.load(f+'.npz'); return np.concatenate([D['ux'].ravel(),D['uy'].ravel()]),D['core']
zA,core=load(A_); zB,_=load(B_)
act=np.ones((2,N,N),bool); act[:,core]=False; act[:,:2]=act[:,-2:]=False; act[:,:,:2]=act[:,:,-2:]=False; act=act.ravel()
zB[~act]=zA[~act]
# relax endpoints at this lam (they should already be minima)
from scipy.optimize import minimize
def relax(z): return minimize(EG,z,jac=True,method='L-BFGS-B',options=dict(maxiter=20000,gtol=1e-10,ftol=1e-15,maxcor=30)).x
zA=relax(zA); zB=relax(zB)
import os
init=os.environ.get('NEB_INIT')
if init:
    P=np.load(init)['Rall']; s_=np.r_[0,np.cumsum(np.linalg.norm(np.diff(P,axis=0),axis=1))]; s_/=s_[-1]
    tt=np.linspace(0,1,M); R=np.array([np.array([np.interp(t,s_,P[:,j]) for j in []]) if False else None for t in tt]) if False else None
    # piecewise-linear resample along arc length
    R=np.empty((M,P.shape[1]))
    for m_,t in enumerate(tt):
        j=min(np.searchsorted(s_,t,side='right')-1,len(P)-2); w=(t-s_[j])/(s_[j+1]-s_[j]); R[m_]=(1-w)*P[j]+w*P[j+1]
    R[0]=zA; R[-1]=zB
else:
    R=np.array([zA+(zB-zA)*t for t in np.linspace(0,1,M)])
V=np.zeros_like(R); dt=0.05; dtmax=0.25; alpha=0.1; npos=0; k=0.02
def forces(R,ci):
    E=np.zeros(M); G=np.zeros_like(R)
    for i in range(M): E[i],G[i]=EG(R[i])
    F=np.zeros_like(R)
    imax=int(np.argmax(E[1:-1]))+1
    for i in range(1,M-1):
        tp=R[i+1]-R[i]; tm=R[i]-R[i-1]
        if E[i+1]>E[i]>E[i-1]: tau=tp
        elif E[i+1]<E[i]<E[i-1]: tau=tm
        else:
            dmax=max(abs(E[i+1]-E[i]),abs(E[i-1]-E[i])); dmin=min(abs(E[i+1]-E[i]),abs(E[i-1]-E[i]))
            tau=tp*dmax+tm*dmin if E[i+1]>E[i-1] else tp*dmin+tm*dmax
        tau=tau/np.linalg.norm(tau); g=-G[i]
        if ci and i==imax: F[i]=g-2*np.dot(g,tau)*tau
        else: F[i]=g-np.dot(g,tau)*tau+k*(np.linalg.norm(tp)-np.linalg.norm(tm))*tau
    F[:,~act]=0
    return E,F,imax
t0=time.time(); log=[]
for it in range(NS):
    ci=it>int(os.environ.get('NEB_CI0',NS//3))
    E,F,imax=forces(R,ci)
    P=np.sum(F*V)
    if P>0:
        vn=np.linalg.norm(V); fn=np.linalg.norm(F)+1e-30
        V=(1-alpha)*V+alpha*vn*F/fn; npos+=1
        if npos>5: dt=min(dt*1.1,dtmax); alpha*=0.99
    else: V[:]=0; dt*=0.5; alpha=0.1; npos=0
    V+=dt*F; R+=dt*V
    fmax=np.abs(F[1:-1]).max()
    if it%100==0 or it==NS-1:
        log.append(dict(it=it,Emax=float(E.max()),E=E.tolist(),fmax=float(fmax)))
        print(f'{tag} it={it} barrier(from A)={E.max()-E[0]:.4f} (from B)={E.max()-E[-1]:.4f} imax={imax} fmax={fmax:.2e} dt={dt:.3f} t={time.time()-t0:.0f}s',flush=True)
    if ci and fmax<2e-5: break
E,F,imax=forces(R,True)
np.savez(f'neb_{tag}.npz',E=E,Rall=R,imax=imax,N=N,x=x)
json.dump(dict(lam=lam,d=d,A=A_,B=B_,E=E.tolist(),imax=imax,log=log),open(f'neb_{tag}.json','w'))
print('DONE',tag,'E profile',np.round(E-E[0],3).tolist())
