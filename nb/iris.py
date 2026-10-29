import os
import numpy as np, sys, time
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','hex'))
from scipy.optimize import minimize
from hexcore import Vhex
s3=np.sqrt(3.); V0=0.05; V3=s3*V0; Lm=12.; th=1.0/Lm*(4*np.pi/s3)/(4*np.pi/s3)
c1=np.array([0.5,1/(2*s3)]); sc=s3/(4*np.pi); lam_s=Lm/(4*np.pi/s3)
def run(C=0.,rh=3.,h=0.4,lam=1.,mu=1.,Lp=48.,Kr=20.,maxit=80000,relax=True):
    N=int(round(2*Lp/h))+1; x=np.linspace(-Lp,Lp,N); X,Y=np.meshgrid(x,x,indexing='ij'); R=np.hypot(X,Y)+1e-12
    # plain twisted moire (same mapping as before): u = c1 + sc*Rot90*(r/lam_s)
    ub_x=c1[0]-sc*Y/lam_s; ub_y=c1[1]+sc*X/lam_s
    # area-deficit (iris-closed hole) field outside the core: u_r = -C/r
    lx=-C*X/R**2; ly=-C*Y/R**2
    core=R<rh; lx[core]=-C*X[core]/rh**2; ly[core]=-C*Y[core]/rh**2
    rim=(R>=rh)&(R<rh+h)
    cm=1.0-(core[:-1,:-1]|core[1:,:-1]|core[:-1,1:]|core[1:,1:]); pm=1.0-core
    fix=np.zeros((N,N),bool); fix[:2]=fix[-2:]=True; fix[:,:2]=fix[:,-2:]=True
    tx_rim=(-C/rh)*X[rim]/R[rim]; ty_rim=(-C/rh)*Y[rim]/R[rim]; nxr=X[rim]/R[rim]; nyr=Y[rim]/R[rim]
    A=h*h
    def grads(f): return ((f[1:,:-1]-f[:-1,:-1])/h,(f[:-1,1:]-f[:-1,:-1])/h,(f[1:,1:]-f[:-1,1:])/h,(f[1:,1:]-f[1:,:-1])/h)
    def adj(gax,gay,gbx,gby):
        o=np.zeros((N,N)); o[1:,:-1]+=gax; o[:-1,:-1]-=gax; o[:-1,1:]+=gay; o[:-1,:-1]-=gay
        o[1:,1:]+=gbx; o[:-1,1:]-=gbx; o[1:,1:]+=gby; o[1:,:-1]-=gby; return o/h
    def EG(z):
        ux,uy=z.reshape(2,N,N); Ux=grads(ux);Uy=grads(uy); E=0.; gu=[[],[]]
        for s in (0,1):
            exx=Ux[2*s];eyy=Uy[2*s+1];exy=0.5*(Ux[2*s+1]+Uy[2*s]);tr=exx+eyy
            E+=0.5*A*np.sum(cm*(mu*(exx**2+eyy**2+2*exy**2)+0.5*lam*tr**2))
            w=0.5*A*cm; gu[0]+=[w*(2*mu*exx+lam*tr),w*2*mu*exy]; gu[1]+=[w*2*mu*exy,w*(2*mu*eyy+lam*tr)]
        G=np.stack([adj(*gu[0]),adj(*gu[1])])
        V,gx,gy=Vhex(ux,uy,V0,V3); E+=A*np.sum(pm*V); G[0]+=A*pm*gx; G[1]+=A*pm*gy
        # rim pulled radially inward by C/rh (relative to the plain moire); tangential free
        dr=(ux[rim]-ub_x[rim]-tx_rim)*nxr+(uy[rim]-ub_y[rim]-ty_rim)*nyr
        E+=0.5*Kr*np.sum(dr**2); G[0][rim]+=Kr*dr*nxr; G[1][rim]+=Kr*dr*nyr
        G[:,fix]=0; G[:,core]=0; return E,G.ravel()
    z0=np.concatenate([(ub_x+lx).ravel(),(ub_y+ly).ravel()])
    if not relax: return dict(x=x,ux=ub_x+lx,uy=ub_y+ly,core=core,E=EG(z0)[0],nit=0,t=0)
    t=time.time()
    r=minimize(EG,z0,jac=True,method='L-BFGS-B',options=dict(maxiter=maxit,maxfun=2*maxit,gtol=1e-10,ftol=1e-15,maxcor=30))
    ux,uy=r.x.reshape(2,N,N)
    return dict(x=x,ux=ux,uy=uy,core=core,E=r.fun,nit=r.nit,t=time.time()-t,ux0=ub_x+lx,uy0=ub_y+ly)
if __name__=="__main__":
    C=float(sys.argv[1]); R_=run(C=C)
    np.savez(f'iris_C{C}.npz',x=R_['x'],ux=R_['ux'],uy=R_['uy'],ux0=R_['ux0'],uy0=R_['uy0'],core=R_['core'],C=C)
    print('C=',C,'E',round(R_['E'],3),'nit',R_['nit'],round(R_['t']),'s',flush=True)
