import os
import numpy as np, sys, time, json
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','hex'))
from scipy.optimize import minimize
from scipy.ndimage import maximum_filter
from hexcore import Vhex
from userfield import w_field, CASES, s3
V0=0.05; V3=s3*V0; Lm=12.; lam_s=Lm/(4*np.pi/s3)     # physical a per user unit
c1=np.array([0.5,1/(2*s3)]); sc=s3/(4*np.pi)          # u = c1 + sc*Rot90*w
def run(case,hole=0.,h=0.4,lam=1.,mu=1.,maxit=60000):
    x0,y0,k,Ds,Cf=CASES[case]
    Lp=30*lam_s; N=int(round(2*Lp/h))+1; x=np.linspace(-Lp,Lp,N); X,Y=np.meshgrid(x,x,indexing='ij')
    wx,wy=w_field(X/lam_s,Y/lam_s,x0,y0,k,Ds,Cf)
    ux=c1[0]-sc*wy; uy=c1[1]+sc*wx
    X0,Y0=x0*lam_s,y0*lam_s
    inpore=((X-X0)**2+((Y-Y0)/k)**2)<hole**2 if hole>0 else np.zeros_like(X,bool)
    cm=1.0-(inpore[:-1,:-1]|inpore[1:,:-1]|inpore[:-1,1:]|inpore[1:,1:])   # cell mask (material)
    pm=1.0-inpore
    fix=np.zeros((N,N),bool); fix[:2]=fix[-2:]=True; fix[:,:2]=fix[:,-2:]=True
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
            w=0.5*A*cm; sxx=w*(2*mu*exx+lam*tr); syy=w*(2*mu*eyy+lam*tr); sxy=w*2*mu*exy
            gu[0]+=[sxx,sxy]; gu[1]+=[sxy,syy]
        G=np.stack([adj(*gu[0]),adj(*gu[1])])
        V,gx,gy=Vhex(ux,uy,V0,V3); E+=A*np.sum(pm*V); G[0]+=A*pm*gx; G[1]+=A*pm*gy
        G[:,fix]=0; G[:,inpore]=0; return E,G.ravel()
    t=time.time()
    r=minimize(EG,np.concatenate([ux.ravel(),uy.ravel()]),jac=True,method='L-BFGS-B',options=dict(maxiter=maxit,maxfun=2*maxit,gtol=1e-10,ftol=1e-15,maxcor=30))
    ux,uy=r.x.reshape(2,N,N)
    return dict(x=x,ux=ux,uy=uy,ux0=c1[0]-sc*wy,uy0=c1[1]+sc*wx,pore=inpore,E=r.fun,nit=r.nit,t=time.time()-t,X0=X0,Y0=Y0,k=k)
def analyse(R,rad):
    x=R['x']; V=Vhex(R['ux'],R['uy'],1.,s3)[0]; V[R['pore']]=0
    pk=(V==maximum_filter(V,size=9))&(V>7.5); pk[:6]=pk[-6:]=False; pk[:,:6]=pk[:,-6:]=False
    I,J=np.nonzero(pk); nodes=[(x[i],x[j]) for i,j in zip(I,J)]
    near=[p for p in nodes if np.hypot(p[0]-R['X0'],(p[1]-R['Y0'])/R['k'])<rad]
    return near,len(nodes)
if __name__=="__main__":
    case=sys.argv[1]; holes=[float(v) for v in sys.argv[2].split(',')]; out=[]
    for hl in holes:
        R=run(case,hole=hl); near,ntot=analyse(R,rad=10.)
        np.savez(f'nb_{case}_h{hl}.npz',**{k:R[k] for k in ['x','ux','uy','ux0','uy0','pore']},X0=R['X0'],Y0=R['Y0'],k=R['k'])
        print(f'{case} pore={hl:4.1f}a  free nodes within 10a of defect: {len(near)} {[(round(a,1),round(b,1)) for a,b in near]}  E={R["E"]:.3f} nit={R["nit"]} {R["t"]:.0f}s',flush=True)
