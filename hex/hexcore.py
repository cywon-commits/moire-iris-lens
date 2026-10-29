import numpy as np, time
from scipy.optimize import minimize
# triangular Bravais lattice a1=(1,0), a2=(1/2,sqrt3/2); reciprocal G1,G2,G3 (sum 0)
s3=np.sqrt(3.)
G=np.array([[2*np.pi,-2*np.pi/s3],[0,4*np.pi/s3],[-2*np.pi,-2*np.pi/s3]])
def Vhex(dx,dy,V0,V3=0.):
    """V=V0*sum(1-cos G.u)+V3*sum sin G.u ; returns V, dV/dux, dV/duy"""
    V=0.;gx=0.;gy=0.
    for g in G:
        ph=g[0]*dx+g[1]*dy; c=np.cos(ph); s=np.sin(ph)
        V=V+V0*(1-c)+V3*s; f=V0*s+V3*c; gx=gx+f*g[0]; gy=gy+f*g[1]
    return V,gx,gy
def mismatch(dx,dy):
    """distance from nearest triangular lattice point"""
    # convert to lattice coords
    n2=dy/(s3/2); n1=dx-0.5*n2
    best=None
    for o1 in (0,1):
        for o2 in (0,1):
            m1=np.floor(n1)+o1; m2=np.floor(n2)+o2
            px=m1+0.5*m2; py=m2*s3/2; d=np.hypot(dx-px,dy-py)
            best=d if best is None else np.minimum(best,d)
    return best
def nodetype(dx,dy):
    """+1 near centroid (a1+a2)/3, -1 near 2(a1+a2)/3 (via sum sin G.u sign)"""
    return sum(np.sin(g[0]*dx+g[1]*dy) for g in G)

def run_periodic(e,c=None,Lx=16*s3,Ly=16.,Nx=112,Ny=64,lam=1.,mu=1.,V0=0.05,V3=0.,rigid0=False,maxit=60000,x0=None,fixmean=True):
    """N layers on rectangular periodic cell. ubar_k=-e_k z + c_k. If rigid0, layer 0 is rigid (no fields)."""
    Nl=len(e); c=c if c is not None else [0j]*Nl
    dx_=Lx/Nx; dy_=Ly/Ny; x=np.arange(Nx)*dx_-Lx/2; y=np.arange(Ny)*dy_-Ly/2
    X,Yg=np.meshgrid(x,y,indexing='ij'); Z=X+1j*Yg
    ub=[-ek*Z+ck for ek,ck in zip(e,c)]
    gb=[((-ek).real,(-1j*ek).real,(-ek).imag,(-1j*ek).imag) for ek in e]
    A=dx_*dy_; R=np.roll
    def grads(f): return ((R(f,-1,0)-f)/dx_,(R(f,-1,1)-f)/dy_,(f-R(f,1,0))/dx_,(f-R(f,1,1))/dy_)
    def adj(a1,a2,b1,b2): return (R(a1,1,0)-a1)/dx_+(R(a2,1,1)-a2)/dy_+(b1-R(b1,-1,0))/dx_+(b2-R(b2,-1,1))/dy_
    k0=1 if rigid0 else 0; nf=Nl-k0
    def EG(z):
        F=z.reshape(nf,2,Nx,Ny); G_=np.zeros_like(F); E=0.
        U=[(ub[k].real,ub[k].imag) if k<k0 else (F[k-k0,0]+ub[k].real,F[k-k0,1]+ub[k].imag) for k in range(Nl)]
        for k in range(k0,Nl):
            Ux=grads(F[k-k0,0]);Uy=grads(F[k-k0,1]); gu=[[],[]]
            for s in (0,1):
                uxx=Ux[2*s]+gb[k][0]; uxy=Ux[2*s+1]+gb[k][1]; uyx=Uy[2*s]+gb[k][2]; uyy=Uy[2*s+1]+gb[k][3]
                exx=uxx;eyy=uyy;exy=0.5*(uxy+uyx);tr=exx+eyy
                E+=0.5*A*np.sum(mu*(exx**2+eyy**2+2*exy**2)+0.5*lam*tr**2)
                sxx=2*mu*exx+lam*tr; syy=2*mu*eyy+lam*tr; sxy=2*mu*exy; w=0.5*A
                gu[0]+=[w*sxx,w*sxy]; gu[1]+=[w*sxy,w*syy]
            G_[k-k0,0]+=adj(*gu[0]); G_[k-k0,1]+=adj(*gu[1])
        for k in range(Nl-1):
            ddx=U[k+1][0]-U[k][0]; ddy=U[k+1][1]-U[k][1]
            V,gx,gy=Vhex(ddx,ddy,V0,V3); E+=A*np.sum(V)
            if k+1>=k0: G_[k+1-k0,0]+=A*gx; G_[k+1-k0,1]+=A*gy
            if k>=k0: G_[k-k0,0]-=A*gx; G_[k-k0,1]-=A*gy
        if fixmean: G_-=G_.mean(axis=(2,3),keepdims=True)
        return E,G_.ravel()
    if x0 is None: x0=np.zeros(nf*2*Nx*Ny)
    if maxit==0: return EG,x0
    t=time.time()
    r=minimize(EG,x0,jac=True,method='L-BFGS-B',options=dict(maxiter=maxit,maxfun=2*maxit,gtol=1e-10,ftol=1e-15,maxcor=30))
    F=r.x.reshape(nf,2,Nx,Ny)
    U=[(ub[k].real,ub[k].imag) if k<k0 else (F[k-k0,0]+ub[k].real,F[k-k0,1]+ub[k].imag) for k in range(Nl)]
    return dict(x=x,y=y,U=U,E=r.fun,nit=r.nit,msg=r.message,t=time.time()-t,xopt=r.x)
