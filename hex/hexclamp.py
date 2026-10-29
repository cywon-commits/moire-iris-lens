import numpy as np, sys, time
from scipy.optimize import minimize
from hexcore import Vhex, s3
def run(shifts,L=12.,d=0.25,lam=1.,mu=1.,V0=0.05,V3=0.,maxit=80000):
    N=int(round(2*L/d))+1; x=np.linspace(-L,L,N); X,Yg=np.meshgrid(x,x,indexing='ij')
    ph=np.mod(np.arctan2(Yg,X),2*np.pi); ns=len(shifts)
    k=np.minimum((ph/(2*np.pi/ns)).astype(int),ns-1)
    S=np.array(shifts); ux=S[k,0]; uy=S[k,1]; ux[N//2,N//2]=0; uy[N//2,N//2]=0
    fix=np.zeros((N,N),bool); fix[:2]=fix[-2:]=True; fix[:,:2]=fix[:,-2:]=True
    A=d*d
    def grads(f): return ((f[1:,:-1]-f[:-1,:-1])/d,(f[:-1,1:]-f[:-1,:-1])/d,(f[1:,1:]-f[:-1,1:])/d,(f[1:,1:]-f[1:,:-1])/d)
    def adj(gax,gay,gbx,gby):
        o=np.zeros((N,N)); o[1:,:-1]+=gax; o[:-1,:-1]-=gax; o[:-1,1:]+=gay; o[:-1,:-1]-=gay
        o[1:,1:]+=gbx; o[:-1,1:]-=gbx; o[1:,1:]+=gby; o[1:,:-1]-=gby; return o/d
    def EG(z):
        ux,uy=z.reshape(2,N,N); Ux=grads(ux);Uy=grads(uy); E=0.; gu=[[],[]]
        for s in (0,1):
            exx=Ux[2*s];eyy=Uy[2*s+1];exy=0.5*(Ux[2*s+1]+Uy[2*s]);tr=exx+eyy
            E+=0.5*A*np.sum(mu*(exx**2+eyy**2+2*exy**2)+0.5*lam*tr**2)
            sxx=2*mu*exx+lam*tr; syy=2*mu*eyy+lam*tr; sxy=2*mu*exy; w=0.5*A
            gu[0]+=[w*sxx,w*sxy]; gu[1]+=[w*sxy,w*syy]
        G=np.stack([adj(*gu[0]),adj(*gu[1])])
        V,gx,gy=Vhex(ux,uy,V0,V3); E+=A*np.sum(V); G[0]+=A*gx; G[1]+=A*gy
        G[:,fix]=0; return E,G.ravel()
    t=time.time()
    r=minimize(EG,np.concatenate([ux.ravel(),uy.ravel()]),jac=True,method='L-BFGS-B',options=dict(maxiter=maxit,maxfun=2*maxit,gtol=1e-10,ftol=1e-15,maxcor=30))
    ux,uy=r.x.reshape(2,N,N); return dict(x=x,ux=ux,uy=uy,ux0=S[k,0],uy0=S[k,1],E=r.fun,nit=r.nit,t=time.time()-t)
if __name__=="__main__":
    tag=sys.argv[1]
    if tag=='six_twist':  sh=[(np.cos(k*np.pi/3-np.pi/3),np.sin(k*np.pi/3-np.pi/3)) for k in range(6)]
    if tag=='six_mix':    sh=[(np.cos(k*np.pi/3),np.sin(k*np.pi/3)) for k in range(6)]
    if tag=='three':      sh=[(0,0),(1,0),(0.5,s3/2)]
    R=run(sh); np.savez(f't2_{tag}.npz',**{k:R[k] for k in ['x','ux','uy','ux0','uy0']})
    print(tag,R['E'],R['nit'],round(R['t']),'s',flush=True)
