import numpy as np, sys, time, json
from scipy.optimize import minimize
from scipy.ndimage import maximum_filter
from hexcore import Vhex, s3
V0=0.05; V3=np.sqrt(3)*V0
c1=np.array([0.5,1/(2*s3)]); c2=np.array([1.,1/s3])
def minima_near(p,rad=0.6):
    out=[]
    for n1 in range(-3,4):
        for n2 in range(-3,4):
            R=np.array([n1+0.5*n2,n2*s3/2])
            for q in (R,R+c2):
                if np.hypot(*(q-p))<rad: out.append(tuple(np.round(q,6)))
    return set(out)
M0=c1; M1=c1+np.array([1.,0.])                      # two adjacent AA maxima
h0=minima_near(M0); h1=minima_near(M1); shared=h0&h1
loop=sorted(h0|h1,key=lambda q: np.arctan2(q[1]-(M0[1]+M1[1])/2,q[0]-(M0[0]+M1[0])/2))
LOOP=np.array(loop)
def run(lam=1.,mu=1.,p=0.,sig=1.5,L=16.,d=0.25,maxit=100000,center=None,cut=0.,bsign=1.):
    N=int(round(2*L/d))+1; x=np.linspace(-L,L,N); X,Yg=np.meshgrid(x,x,indexing='ij')
    mid=(M0+M1)/2; angs=np.arctan2(LOOP[:,1]-mid[1],LOOP[:,0]-mid[0])
    # sector boundaries halfway between consecutive minimum angles
    a=np.unwrap(angs); bnd=(a+np.roll(a,-1)+2*np.pi*(np.arange(len(a))==len(a)-1))/2
    ph=np.arctan2(Yg,X); k=np.zeros(X.shape,int)
    for i in range(len(a)):
        lo=bnd[i-1]-(2*np.pi if i==0 else 0); hi=bnd[i]
        sel=((ph-lo)%(2*np.pi))<((hi-lo)%(2*np.pi)); k[sel]=i
    ux=LOOP[k,0]; uy=LOOP[k,1]
    g=np.exp(-(X**2+Yg**2)/(2*sig**2)); w=1-p*g
    fix=np.zeros((N,N),bool); fix[:2]=fix[-2:]=True; fix[:,:2]=fix[:,-2:]=True
    A=d*d
    # eigen-slip: u_x jumps by bsign*1 (= a1) across x=0 for |y|<cut/2  (extra half-row segment)
    i0=N//2-1; ys=x; segA=np.zeros((N-1,N-1)); segB=np.zeros((N-1,N-1))
    jj=np.nonzero(np.abs(ys)<cut/2)[0]
    segA[i0,jj[jj<N-1]]=bsign/d          # stencil A: (f[i+1]-f[i]) at i=i0, rows j
    segB[i0,jj[jj>=1]-1]=bsign/d         # stencil B index (i-1,j-1) for i=i0+1, j
    seg=[segA,segB]
    def grads(f): return ((f[1:,:-1]-f[:-1,:-1])/d,(f[:-1,1:]-f[:-1,:-1])/d,(f[1:,1:]-f[:-1,1:])/d,(f[1:,1:]-f[1:,:-1])/d)
    def adj(gax,gay,gbx,gby):
        o=np.zeros((N,N)); o[1:,:-1]+=gax; o[:-1,:-1]-=gax; o[:-1,1:]+=gay; o[:-1,:-1]-=gay
        o[1:,1:]+=gbx; o[:-1,1:]-=gbx; o[1:,1:]+=gby; o[1:,:-1]-=gby; return o/d
    def EG(z):
        ux,uy=z.reshape(2,N,N); Ux=grads(ux);Uy=grads(uy); E=0.; gu=[[],[]]
        for s in (0,1):
            exx=Ux[2*s]-seg[s];eyy=Uy[2*s+1];exy=0.5*(Ux[2*s+1]+Uy[2*s]);tr=exx+eyy
            E+=0.5*A*np.sum(mu*(exx**2+eyy**2+2*exy**2)+0.5*lam*tr**2)
            sxx=2*mu*exx+lam*tr; syy=2*mu*eyy+lam*tr; sxy=2*mu*exy; ww=0.5*A
            gu[0]+=[ww*sxx,ww*sxy]; gu[1]+=[ww*sxy,ww*syy]
        G=np.stack([adj(*gu[0]),adj(*gu[1])])
        V,gx,gy=Vhex(ux,uy,V0,V3); E+=A*np.sum(w*V); G[0]+=A*w*gx; G[1]+=A*w*gy
        G[:,fix]=0; return E,G.ravel()
    t=time.time()
    r=minimize(EG,np.concatenate([ux.ravel(),uy.ravel()]),jac=True,method='L-BFGS-B',options=dict(maxiter=maxit,maxfun=2*maxit,gtol=1e-10,ftol=1e-15,maxcor=30))
    ux,uy=r.x.reshape(2,N,N)
    return dict(x=x,ux=ux,uy=uy,E=r.fun,nit=r.nit,t=time.time()-t)
def nodes(R,thr=7.0):
    x=R['x']; V=Vhex(R['ux'],R['uy'],1.,np.sqrt(3.))[0]
    pk=(V==maximum_filter(V,size=9))&(V>thr); n=len(x); pk[:8]=pk[-8:]=False; pk[:,:8]=pk[:,-8:]=False
    I,J=np.nonzero(pk); return [(x[i],x[j],V[i,j]) for i,j in zip(I,J)]
if __name__=="__main__":
    print('loop minima (u-space):',len(LOOP),' shared:',sorted(shared))
    mode=sys.argv[1]; vals=[float(v) for v in sys.argv[2].split(',')]; res=[]
    bs=float(sys.argv[3]) if len(sys.argv)>3 else 1.
    for v in vals:
        R=run(lam=v) if mode=='lam' else (run(p=v) if mode=='pin' else run(cut=v,bsign=bs))
        nd=nodes(R); D=max([np.hypot(a[0]-b[0],a[1]-b[1]) for a in nd for b in nd],default=0.)
        res.append(dict(v=v,D=D,nodes=nd,E=R['E']))
        np.savez(f'ten_{mode}{"" if mode!="cut" else ("p" if bs>0 else "m")}_{v}.npz',x=R['x'],ux=R['ux'],uy=R['uy'])
        print(f'{mode}={v:6.2f}  nodes={len(nd)}  D={D:6.2f}  Vpk={[round(q[2],2) for q in nd]}  pos={[(round(q[0],1),round(q[1],1)) for q in nd]}  nit={R["nit"]} {R["t"]:.0f}s',flush=True)
    json.dump(res,open(f'ten_{mode}_{sys.argv[2]}_{bs}.json','w'))
