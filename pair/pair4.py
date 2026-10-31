import numpy as np, sys, time, json
import os; sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','hex'))
from scipy.optimize import minimize
from hexcore import Vhex
s3=np.sqrt(3.); c1=np.array([0.5,1/(2*s3)])
def run(defects,Lm=30.,V0=0.005,alpha=90.,h=0.45,Lp=72.,lam=1.,mu=1.,rh=2.4,Kr=200.,maxit=60000,x0=None,mirror=False,symm=None,seed=None,seed_amp=0.05,seed_len=8.,want_EG=False,Ua=0j,Ub=0j,incl=None):
    """defects: list of (x,y,C). area deficit 2*pi*C enforced through mean inward rim displacement; rim shape free"""
    V3=s3*V0; th=1/Lm; N=int(round(2*Lp/h))+1; x=np.linspace(-Lp,Lp,N); X,Y=np.meshgrid(x,x,indexing='ij'); Z=X+1j*Y
    kap=th*np.exp(1j*np.radians(alpha)); W=kap*Z-kap*(Ua*Z+Ub*np.conj(Z)); ub=c1[0]+W.real, c1[1]+W.imag
    sxI=0.;syI=0.;gI=None
    if incl is not None:   # moire eigen-distortion U*=eps*(z-z0) inside disk: energy evaluated on u+s, s=kap*eps*(z-z0)[disk]
        xi,yi,ri,epsI=incl; zz=Z-(xi+1j*yi); dsk=np.abs(zz)<ri; sI=kap*epsI*zz*dsk; sxI=sI.real; syI=sI.imag
        Xc=0.5*(X[1:,:-1]+X[:-1,:-1])+0.5*(X[1,0]-X[0,0])*0; 
        zc=(X[:-1,:-1]+h/2-xi)+1j*(Y[:-1,:-1]+h/2-yi); cdk=np.abs(zc)<ri; gk=kap*epsI   # d/dx(kap eps z)=kap eps ; d/dy = i kap eps
        gI=(gk.real*cdk,(1j*gk).real*cdk,gk.imag*cdk,(1j*gk).imag*cdk)   # dux/dx, dux/dy, duy/dx, duy/dy
    core=np.zeros_like(X,bool); rims=[]; lx=np.zeros_like(X); ly=np.zeros_like(X)
    for (xd,yd,C) in defects:
        R=np.hypot(X-xd,Y-yd)+1e-12; core|=R<rh
        C=complex(C); w=-C*((X-xd)+1j*(Y-yd))/np.maximum(R,rh)**2; lx+=w.real; ly+=w.imag
        rim=(R>=rh)&(R<rh+h); rims.append((rim,(X[rim]-xd)/R[rim],(Y[rim]-yd)/R[rim],-C.real/rh,-C.imag/rh))
    cm=1.0-(core[:-1,:-1]|core[1:,:-1]|core[:-1,1:]|core[1:,1:]); pm=1.0-core
    fix=np.zeros((N,N),bool); fix[:2]=fix[-2:]=True; fix[:,:2]=fix[:,-2:]=True
    A=h*h
    def grads(f): return ((f[1:,:-1]-f[:-1,:-1])/h,(f[:-1,1:]-f[:-1,:-1])/h,(f[1:,1:]-f[:-1,1:])/h,(f[1:,1:]-f[1:,:-1])/h)
    def adj(gax,gay,gbx,gby):
        o=np.zeros((N,N)); o[1:,:-1]+=gax; o[:-1,:-1]-=gax; o[:-1,1:]+=gay; o[:-1,:-1]-=gay
        o[1:,1:]+=gbx; o[:-1,1:]-=gbx; o[1:,1:]+=gby; o[1:,:-1]-=gby; return o/h
    def EG(z):
        ux,uy=z.reshape(2,N,N); Ux=grads(ux);Uy=grads(uy); E=0.; gu=[[],[]]; Eel=0.
        if gI is not None:
            Ux=tuple(Ux[k]+gI[k%2] for k in range(4)); Uy=tuple(Uy[k]+gI[2+k%2] for k in range(4))
        for s in (0,1):
            exx=Ux[2*s];eyy=Uy[2*s+1];exy=0.5*(Ux[2*s+1]+Uy[2*s]);tr=exx+eyy
            e=0.5*A*np.sum(cm*(mu*(exx**2+eyy**2+2*exy**2)+0.5*lam*tr**2)); E+=e
            w=0.5*A*cm; gu[0]+=[w*(2*mu*exx+lam*tr),w*2*mu*exy]; gu[1]+=[w*2*mu*exy,w*(2*mu*eyy+lam*tr)]
        G=np.stack([adj(*gu[0]),adj(*gu[1])])
        if V0>0:
            V,gx,gy=Vhex(ux+sxI,uy+syI,V0,V3); E+=A*np.sum(pm*V); G[0]+=A*pm*gx; G[1]+=A*pm*gy
        for rim,nx,ny,tgt,tgt_t in rims:
            n=rim.sum(); dr=np.mean((ux[rim]-ub[0][rim])*nx+(uy[rim]-ub[1][rim])*ny)-tgt
            E+=0.5*Kr*n*dr**2; G[0][rim]+=Kr*dr*nx; G[1][rim]+=Kr*dr*ny
            if tgt_t!=0.:
                dt_=np.mean(-(ux[rim]-ub[0][rim])*ny+(uy[rim]-ub[1][rim])*nx)-tgt_t
                E+=0.5*Kr*n*dt_**2; G[0][rim]+=-Kr*dt_*ny; G[1][rim]+=Kr*dt_*nx
        G[:,fix]=0; G[:,core]=0
        if mirror:   # project onto configurations symmetric under y->-y (mirror through AA): u_x even, u_y-c1_y odd
            G=np.stack([0.5*(G[0]+G[0][:,::-1]),0.5*(G[1]-G[1][:,::-1])])
        if symm is not None: G=proj(G)
        return E,G.ravel()
    def proj(G):
        if symm=='xm': return np.stack([0.5*(G[0]-G[0][::-1,:]),0.5*(G[1]+G[1][::-1,:])])
        if symm=='c2': return np.stack([0.5*(G[0]-G[0][::-1,::-1]),0.5*(G[1]-G[1][::-1,::-1])])
        return G
    if want_EG: return EG,N,x
    if x0 is None:
        sx=np.zeros_like(X); sy=np.zeros_like(X)
        if seed is not None:
            for (xd,yd,C),sg in zip(defects,seed):
                dx_,dy_=X-xd,Y-yd; R=np.hypot(dx_,dy_)+1e-9; f=sg*seed_amp*np.exp(-R/seed_len)
                sx+=-f*dy_/R; sy+=f*dx_/R
        tx=lx+sx; ty=ly+sy
        if symm is not None:
            T=proj(np.stack([tx,ty])); tx,ty=T[0],T[1]
        x0=np.concatenate([(ub[0]+tx).ravel(),(ub[1]+ty).ravel()])
    t=time.time()
    r=minimize(EG,x0,jac=True,method='L-BFGS-B',options=dict(maxiter=maxit,maxfun=2*maxit,gtol=1e-10,ftol=1e-15,maxcor=30))
    ux,uy=r.x.reshape(2,N,N)
    return dict(EG=EG,x=x,ux=ux,uy=uy,core=core,E=float(r.fun),nit=r.nit,t=time.time()-t)
