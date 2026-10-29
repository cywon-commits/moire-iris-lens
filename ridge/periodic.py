import numpy as np, sys, time
from scipy.optimize import minimize
# periodic L x L cell, a=1. u = ubar + ut,  ubar = -(1+i) z / Lm  (Lm: moire scale, rotation+contraction a/Lm)
# cell L=Lm is commensurate: ubar(r+L e) - ubar(r) = lattice vector.
def run(L=16.,d=0.25,lam=1.,mu=1.,V0=0.05,kap=0.1,Kz=0.1,d1=0.,maxit=80000,seed=0,ut0=None,h0=None):
    N=int(round(L/d)); x=np.arange(N)*d-L/2; X,Yg=np.meshgrid(x,x,indexing='ij')
    s=1./L
    ubx=-(X-Yg)*s; uby=-(X+Yg)*s
    gbx=(-s, s); gby=(-s,-s)          # (d/dx, d/dy) of ubar_x, ubar_y
    rng=np.random.default_rng(seed)
    ut=np.zeros((2,N,N)) if ut0 is None else ut0
    h=1e-3*rng.standard_normal((N,N)) if h0 is None else h0
    p=2*np.pi; A=d*d; R=np.roll
    def grads(f):
        return ((R(f,-1,0)-f)/d,(R(f,-1,1)-f)/d,(f-R(f,1,0))/d,(f-R(f,1,1))/d)
    def adj(gax,gay,gbx_,gby_):
        return ((R(gax,1,0)-gax)+(R(gay,1,1)-gay)+(gbx_-R(gbx_,-1,0))+(gby_-R(gby_,-1,1)))/d
    def lapl(f): return (R(f,1,0)+R(f,-1,0)+R(f,1,1)+R(f,-1,1)-4*f)/d**2
    def EG(z):
        tx,ty,h=z.reshape(3,N,N)
        Ux=grads(tx);Uy=grads(ty);Hd=grads(h)
        E=0.; gu=[[],[],[]]
        for k in (0,1):
            uxx=Ux[2*k]+gbx[0]; uxy=Ux[2*k+1]+gbx[1]; uyx=Uy[2*k]+gby[0]; uyy=Uy[2*k+1]+gby[1]
            hx,hy=Hd[2*k],Hd[2*k+1]
            exx=uxx+0.5*hx**2; eyy=uyy+0.5*hy**2; exy=0.5*(uxy+uyx)+0.5*hx*hy; tr=exx+eyy
            E+=0.5*A*np.sum(mu*(exx**2+eyy**2+2*exy**2)+0.5*lam*tr**2)
            sxx=2*mu*exx+lam*tr; syy=2*mu*eyy+lam*tr; sxy=2*mu*exy; w=0.5*A
            gu[0]+=[w*sxx,w*sxy]; gu[1]+=[w*sxy,w*syy]; gu[2]+=[w*(sxx*hx+sxy*hy),w*(syy*hy+sxy*hx)]
        G=[adj(*gu[k]) for k in range(3)]
        lp=lapl(h); E+=A*np.sum(0.5*kap*lp**2); G[2]+=A*kap*lapl(lp)
        ux=tx+ubx; uy=ty+uby
        dd=0.25*d1*(2-np.cos(p*ux)-np.cos(p*uy))
        r_=h-dd
        E+=A*np.sum(0.5*Kz*r_**2+V0*(2-np.cos(p*ux)-np.cos(p*uy)))
        G[0]+=A*(V0*p*np.sin(p*ux)-Kz*r_*0.25*d1*p*np.sin(p*ux))
        G[1]+=A*(V0*p*np.sin(p*uy)-Kz*r_*0.25*d1*p*np.sin(p*uy))
        G[2]+=A*Kz*r_
        return E,np.stack(G).ravel()
    z0=np.concatenate([ut.ravel(),h.ravel()])
    if maxit==0: return EG,z0
    t=time.time()
    r=minimize(EG,z0,jac=True,method='L-BFGS-B',options=dict(maxiter=maxit,maxfun=2*maxit,gtol=1e-10,ftol=1e-15,maxcor=30))
    tx,ty,h=r.x.reshape(3,N,N)
    return dict(x=x,ux=tx+ubx,uy=ty+uby,tx=tx,ty=ty,h=h,E=r.fun,nit=r.nit,msg=r.message,t=time.time()-t)
if __name__=="__main__":
    lb,Lam,d1,tag=float(sys.argv[1]),float(sys.argv[2]),float(sys.argv[3]),sys.argv[4]
    Y,V0=3.,0.05; P=np.sqrt(Lam*Y*V0)
    R_=run(V0=V0,kap=P*lb**2,Kz=P/lb**2,d1=d1)
    np.savez(f'per_{tag}.npz',**{k:R_[k] for k in ['x','ux','uy','h']},lb=lb,Lam=Lam,d1=d1,E=R_['E'])
    print(tag,lb,Lam,d1,'hmax',round(R_['h'].max(),4),'hmin',round(R_['h'].min(),4),'E',R_['E'],'nit',R_['nit'],R_['msg'],round(R_['t']),'s',flush=True)
