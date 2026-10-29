import os
import numpy as np, sys, time
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','hex'))
from scipy.optimize import minimize
from hexcore import Vhex
from userfield import w_field, CASES, s3
V0=0.05; V3=s3*V0; Lm=12.; lam_s=Lm/(4*np.pi/s3)
c1=np.array([0.5,1/(2*s3)]); sc=s3/(4*np.pi)
def run(H0=0.,rho=4.,xc=0.,yc=0.,case=None,h=0.4,lam=1.,mu=1.,kap=0.02,Gam=0.6,dg=1.0,d1=0.3,Kc=50.,Ko=50.,Lbox=30.,maxit=80000,x0=None,ret_EG=False,hnoise=2e-3):
    Lp=Lbox*lam_s; N=int(round(2*Lp/h))+1; x=np.linspace(-Lp,Lp,N); X,Y=np.meshgrid(x,x,indexing='ij')
    if case is None: wx,wy=X/lam_s,Y/lam_s
    else:
        cx,cy,k,Ds,Cf=CASES[case]; wx,wy=w_field(X/lam_s,Y/lam_s,cx,cy,k,Ds,Cf)
    ux0=c1[0]-sc*wy; uy0=c1[1]+sc*wx
    R2=(X-xc)**2+(Y-yc)**2; f=H0*np.maximum(1-R2/rho**2,0)
    fix=np.zeros((N,N),bool); fix[:3]=fix[-3:]=True; fix[:,:3]=fix[:,-3:]=True
    A=h*h; R=np.roll
    def grads(F): return ((F[1:,:-1]-F[:-1,:-1])/h,(F[:-1,1:]-F[:-1,:-1])/h,(F[1:,1:]-F[:-1,1:])/h,(F[1:,1:]-F[1:,:-1])/h)
    def adj(gax,gay,gbx,gby):
        o=np.zeros((N,N)); o[1:,:-1]+=gax; o[:-1,:-1]-=gax; o[:-1,1:]+=gay; o[:-1,:-1]-=gay
        o[1:,1:]+=gbx; o[:-1,1:]-=gbx; o[1:,1:]+=gby; o[1:,:-1]-=gby; return o/h
    def lapl(F):
        o=np.zeros_like(F); o[1:-1,1:-1]=(F[2:,1:-1]+F[:-2,1:-1]+F[1:-1,2:]+F[1:-1,:-2]-4*F[1:-1,1:-1])/h**2; return o
    def EG(z):
        ux,uy,hh=z.reshape(3,N,N); Ux=grads(ux);Uy=grads(uy);Hd=grads(hh); E=0.; gu=[[],[],[]]
        for s in (0,1):
            hx,hy=Hd[2*s],Hd[2*s+1]
            exx=Ux[2*s]+0.5*hx**2; eyy=Uy[2*s+1]+0.5*hy**2; exy=0.5*(Ux[2*s+1]+Uy[2*s])+0.5*hx*hy; tr=exx+eyy
            E+=0.5*A*np.sum(mu*(exx**2+eyy**2+2*exy**2)+0.5*lam*tr**2)
            sxx=2*mu*exx+lam*tr; syy=2*mu*eyy+lam*tr; sxy=2*mu*exy; w=0.5*A
            gu[0]+=[w*sxx,w*sxy]; gu[1]+=[w*sxy,w*syy]; gu[2]+=[w*(sxx*hx+sxy*hy),w*(syy*hy+sxy*hx)]
        G=[adj(*gu[k]) for k in range(3)]
        lp=lapl(hh); E+=A*np.sum(0.5*kap*lp**2); G[2]+=A*kap*lapl(lp)
        V,gx,gy=Vhex(ux,uy,V0,V3); cd=d1/(9*V0)
        dd=cd*V; hp=hh-dd; g=np.exp(-(hp/dg)**2); gp=-2*hp/dg**2*g
        E+=A*np.sum(g*V+Gam*(1-g))
        G[0]+=A*(g*gx-(V-Gam)*gp*cd*gx); G[1]+=A*(g*gy-(V-Gam)*gp*cd*gy); G[2]+=A*(V-Gam)*gp
        hm=np.minimum(hh,0); E+=A*np.sum(0.5*Kc*hm**2); G[2]+=A*Kc*hm
        ob=np.maximum(f-hh,0); E+=A*np.sum(0.5*Ko*ob**2); G[2]-=A*Ko*ob
        G=np.stack(G); G[:,fix]=0; return E,G.ravel()
    if x0 is None:
        hinit=np.maximum(f,0)*1.0+hnoise*np.random.default_rng(7).standard_normal(f.shape)
        x0=np.concatenate([ux0.ravel(),uy0.ravel(),hinit.ravel()])
    if ret_EG: return EG,x0
    t=time.time()
    r=minimize(EG,x0,jac=True,method='L-BFGS-B',options=dict(maxiter=maxit,maxfun=2*maxit,gtol=1e-10,ftol=1e-15,maxcor=30))
    ux,uy,hh=r.x.reshape(3,N,N)
    return dict(x=x,ux=ux,uy=uy,h=hh,f=f,E=r.fun,nit=r.nit,t=time.time()-t,xopt=r.x,xc=xc,yc=yc)
if __name__=="__main__":
    EG,z0=run(H0=2.,Lbox=3,ret_EG=True)
    z0=z0+0.05*np.random.default_rng(0).standard_normal(z0.size)
    z0[2*(len(z0)//3):]=np.abs(z0[2*(len(z0)//3):])*0.5
    E,Gr=EG(z0)
    for k in np.random.default_rng(1).choice(np.nonzero(Gr)[0],10,replace=False):
        e=np.zeros_like(z0); e[k]=1e-6; print(k,Gr[k],(EG(z0+e)[0]-EG(z0-e)[0])/2e-6)
