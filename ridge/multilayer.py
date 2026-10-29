import numpy as np, time
from scipy.optimize import minimize
# N layers, periodic L x L cell, a=1. Layer k: u_k = ubar_k + ut_k, ubar_k = -e_k z + c_k (complex).
# Commensurability: e_k*L must be a Gaussian integer.
def run(e,c=None,L=16.,d=0.25,lam=1.,mu=1.,V0=0.05,kap=0.1,Kz=0.1,d1=0.,zfree=False,fixmean=True,Kext=None,
        maxit=60000,seed=0,x0=None,hseed=1e-3):
    Nl=len(e); c=c if c is not None else [0j]*Nl
    N=int(round(L/d)); x=np.arange(N)*d-L/2; X,Yg=np.meshgrid(x,x,indexing='ij'); Z=X+1j*Yg
    ub=[-ek*Z+ck for ek,ck in zip(e,c)]; ubx=[u.real for u in ub]; uby=[u.imag for u in ub]
    # grads of ubar: d/dx(-e z) = -e ; d/dy(-e z) = -i e
    gb=[((-ek).real,(-1j*ek).real,(-ek).imag,(-1j*ek).imag) for ek in e]  # (dux/dx,dux/dy,duy/dx,duy/dy)
    p=2*np.pi; A=d*d; R=np.roll
    def grads(f): return ((R(f,-1,0)-f)/d,(R(f,-1,1)-f)/d,(f-R(f,1,0))/d,(f-R(f,1,1))/d)
    def adj(a1,a2,b1,b2): return ((R(a1,1,0)-a1)+(R(a2,1,1)-a2)+(b1-R(b1,-1,0))+(b2-R(b2,-1,1)))/d
    def lapl(f): return (R(f,1,0)+R(f,-1,0)+R(f,1,1)+R(f,-1,1)-4*f)/d**2
    nv=3 if zfree else 2
    def EG(z):
        F=z.reshape(Nl,nv,N,N); G=np.zeros_like(F); E=0.
        for k in range(Nl):
            tx,ty=F[k,0],F[k,1]; h=F[k,2] if zfree else None
            Ux=grads(tx);Uy=grads(ty); Hd=grads(h) if zfree else [0]*4
            gu=[[],[],[]]
            for s in (0,1):
                uxx=Ux[2*s]+gb[k][0]; uxy=Ux[2*s+1]+gb[k][1]; uyx=Uy[2*s]+gb[k][2]; uyy=Uy[2*s+1]+gb[k][3]
                hx,hy=Hd[2*s],Hd[2*s+1]
                exx=uxx+0.5*hx**2*zfree; eyy=uyy+0.5*hy**2*zfree; exy=0.5*(uxy+uyx)+0.5*hx*hy*zfree; tr=exx+eyy
                E+=0.5*A*np.sum(mu*(exx**2+eyy**2+2*exy**2)+0.5*lam*tr**2)
                sxx=2*mu*exx+lam*tr; syy=2*mu*eyy+lam*tr; sxy=2*mu*exy; w=0.5*A
                gu[0]+=[w*sxx,w*sxy]; gu[1]+=[w*sxy,w*syy]
                if zfree: gu[2]+=[w*(sxx*hx+sxy*hy),w*(syy*hy+sxy*hx)]
            G[k,0]+=adj(*gu[0]); G[k,1]+=adj(*gu[1])
            if zfree:
                G[k,2]+=adj(*gu[2]); lp=lapl(h); E+=A*np.sum(0.5*kap*lp**2); G[k,2]+=A*kap*lapl(lp)
        for k in range(Nl-1):
            dx=(F[k+1,0]+ubx[k+1])-(F[k,0]+ubx[k]); dy=(F[k+1,1]+uby[k+1])-(F[k,1]+uby[k])
            E+=A*np.sum(V0*(2-np.cos(p*dx)-np.cos(p*dy)))
            gx=A*V0*p*np.sin(p*dx); gy=A*V0*p*np.sin(p*dy)
            if zfree:
                dd=0.25*d1*(2-np.cos(p*dx)-np.cos(p*dy)); r_=F[k+1,2]-F[k,2]-dd
                E+=A*np.sum(0.5*Kz*r_**2)
                G[k+1,2]+=A*Kz*r_; G[k,2]-=A*Kz*r_
                gx=gx-A*Kz*r_*0.25*d1*p*np.sin(p*dx); gy=gy-A*Kz*r_*0.25*d1*p*np.sin(p*dy)
            G[k+1,0]+=gx; G[k,0]-=gx; G[k+1,1]+=gy; G[k,1]-=gy
        if zfree and Kext:
            for k in (0,Nl-1):
                E+=A*np.sum(0.5*Kext*F[k,2]**2); G[k,2]+=A*Kext*F[k,2]
        if fixmean: G[:,:2]-=G[:,:2].mean(axis=(2,3),keepdims=True)   # keep mean in-plane displacement of each layer fixed
        return E,G.ravel()
    rng=np.random.default_rng(seed)
    if x0 is None:
        x0=np.zeros((Nl,nv,N,N))
        if zfree: x0[:,2]=hseed*rng.standard_normal((Nl,N,N))
        x0=x0.ravel()
    if maxit==0: return EG,x0
    t=time.time()
    r=minimize(EG,x0,jac=True,method='L-BFGS-B',options=dict(maxiter=maxit,maxfun=2*maxit,gtol=1e-10,ftol=1e-15,maxcor=30))
    F=r.x.reshape(Nl,nv,N,N)
    U=[(F[k,0]+ubx[k],F[k,1]+uby[k]) for k in range(Nl)]
    H=[F[k,2] for k in range(Nl)] if zfree else None
    return dict(x=x,U=U,H=H,E=r.fun,nit=r.nit,msg=r.message,t=time.time()-t,xopt=r.x)
