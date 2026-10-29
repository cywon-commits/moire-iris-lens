import numpy as np, sys, time
from scipy.optimize import minimize
def run(L=10.,d=0.25,lam=1.,mu=1.,V0=0.05,kap=0.1,Kz=0.1,maxit=60000,seed=0,mode="user"):
    N=int(round(2*L/d))+1; x=np.linspace(-L,L,N); X,Yg=np.meshgrid(x,x,indexing='ij')
    q1=(X>0)&(Yg>=0); q2=(X<=0)&(Yg>0); q3=(X<0)&(Yg<=0); q4=(X>=0)&(Yg<0)
    ux=np.where(q2,1.,0.)+np.where(q4,-1.,0.); uy=np.where(q1,-1.,0.)+np.where(q3,1.,0.)
    if mode=="n2": ux=2*ux; uy=2*uy
    if mode=="oct":
        ph=np.arctan2(Yg,X); k=np.floor((ph+np.pi/8)/(np.pi/4)).astype(int)%8
        tab=np.array([(-1,-1),(0,-1),(1,-1),(1,0),(1,1),(0,1),(-1,1),(-1,0)],float)
        tab=np.roll(tab,0,axis=0)
        ux=tab[k,0]; uy=tab[k,1]
    if mode=="rect6":
        verts=np.array([(0,0),(1,0),(2,0),(2,1),(1,1),(0,1)],float); mid=np.array([1.,0.5])
        a=np.arctan2(verts[:,1]-mid[1],verts[:,0]-mid[0]); o=np.argsort(a); verts=verts[o]; a=a[o]
        b=(a+np.roll(a,-1)+2*np.pi*(np.arange(6)==5))/2; ph=np.arctan2(Yg,X); k=np.zeros(X.shape,int)
        for i in range(6):
            lo=b[i-1]-(2*np.pi if i==0 else 0); hi=b[i]; sel=((ph-lo)%(2*np.pi))<((hi-lo)%(2*np.pi)); k[sel]=i
        ux=verts[k,0]; uy=verts[k,1]
    if mode=="twist": ux=np.where(q1|q2,-1.,0.); uy=np.where(q1|q4,1.,0.)
    rng=np.random.default_rng(seed)
    h=0.05*np.exp(-(X**2+Yg**2)/4)+1e-3*rng.standard_normal(X.shape)
    fix=np.zeros((N,N),bool); fix[:2]=fix[-2:]=True; fix[:,:2]=fix[:,-2:]=True
    h[fix]=0; p=2*np.pi; A=d*d
    def grads(f):
        return ((f[1:,:-1]-f[:-1,:-1])/d,(f[:-1,1:]-f[:-1,:-1])/d,
                (f[1:,1:]-f[:-1,1:])/d,(f[1:,1:]-f[1:,:-1])/d)
    def adj(gax,gay,gbx,gby):
        o=np.zeros((N,N))
        o[1:,:-1]+=gax; o[:-1,:-1]-=gax; o[:-1,1:]+=gay; o[:-1,:-1]-=gay
        o[1:,1:]+=gbx; o[:-1,1:]-=gbx; o[1:,1:]+=gby; o[1:,:-1]-=gby
        return o/d
    def EG(z):
        ux,uy,h=z.reshape(3,N,N)
        Ux=grads(ux);Uy=grads(uy);Hd=grads(h)
        E=0; g=[[None]*2 for _ in range(2)]; gu=[[],[],[]]
        for s in (0,1):
            uxx,uxy=Ux[2*s],Ux[2*s+1]; uyx,uyy=Uy[2*s],Uy[2*s+1]; hx,hy=Hd[2*s],Hd[2*s+1]
            exx=uxx+0.5*hx**2; eyy=uyy+0.5*hy**2; exy=0.5*(uxy+uyx)+0.5*hx*hy
            tr=exx+eyy
            E+=0.5*A*np.sum(mu*(exx**2+eyy**2+2*exy**2)+0.5*lam*tr**2)
            sxx=2*mu*exx+lam*tr; syy=2*mu*eyy+lam*tr; sxy=2*mu*exy
            w=0.5*A
            gu[0]+= [w*sxx,w*sxy]; gu[1]+=[w*sxy,w*syy]
            gu[2]+=[w*(sxx*hx+sxy*hy),w*(syy*hy+sxy*hx)]
        G=[adj(*gu[k]) for k in range(3)]
        lap=(h[2:,1:-1]+h[:-2,1:-1]+h[1:-1,2:]+h[1:-1,:-2]-4*h[1:-1,1:-1])/d**2
        E+=A*np.sum(0.5*kap*lap**2); gl=A*kap*lap/d**2
        Gl=np.zeros((N,N)); Gl[2:,1:-1]+=gl;Gl[:-2,1:-1]+=gl;Gl[1:-1,2:]+=gl;Gl[1:-1,:-2]+=gl;Gl[1:-1,1:-1]-=4*gl
        G[2]+=Gl
        E+=A*np.sum(0.5*Kz*h**2+V0*(2-np.cos(p*ux)-np.cos(p*uy)))
        G[0]+=A*V0*p*np.sin(p*ux); G[1]+=A*V0*p*np.sin(p*uy); G[2]+=A*Kz*h
        G=np.stack(G); G[:,fix]=0
        return E,G.ravel()
    t=time.time()
    r=minimize(EG,np.concatenate([ux.ravel(),uy.ravel(),h.ravel()]),jac=True,method='L-BFGS-B',
               options=dict(maxiter=maxit,maxfun=2*maxit,gtol=1e-10,ftol=1e-14,maxcor=30))
    ux,uy,h=r.x.reshape(3,N,N)
    return dict(x=x,ux=ux,uy=uy,h=h,E=r.fun,nit=r.nit,msg=r.message,t=time.time()-t)
if __name__=="__main__":
    lb=float(sys.argv[1]); Lam=float(sys.argv[2]); tag=sys.argv[3]
    Y,V0=3.,0.05; P=np.sqrt(Lam*Y*V0)
    R=run(V0=V0,kap=P*lb**2,Kz=P/lb**2,mode=(sys.argv[4] if len(sys.argv)>4 else "user"))
    np.savez(f'r2d_{tag}.npz',**{k:R[k] for k in ['x','ux','uy','h']},lb=lb,Lam=Lam)
    print(tag,lb,Lam,'Hmax',R['h'].max(),'Hmin',R['h'].min(),'nit',R['nit'],R['msg'],round(R['t']),'s')
