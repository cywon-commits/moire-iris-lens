import numpy as np, time, sys
from scipy.optimize import minimize
from hexcore import Vhex, s3, mismatch
# bilayer: bottom rigid, top elastic; rigid relative map ubar = -A r ; periodic cell Lx=N*sqrt3, Ly=N
def run(A,N=32,h=0.25,lam=1.,mu=1.,V0=0.05,V3=np.sqrt(3)*0.05,maxit=80000,c=(0.,0.),sym=False,x0=None):
    Lx=N*s3; Ly=float(N); Nx=int(round(Lx/h)); Ny=int(round(Ly/h)); dx_=Lx/Nx; dy_=Ly/Ny
    x=np.arange(Nx)*dx_; y=np.arange(Ny)*dy_; X,Yg=np.meshgrid(x,y,indexing='ij')
    A=np.array(A,float); ubx=-(A[0,0]*X+A[0,1]*Yg)+c[0]; uby=-(A[1,0]*X+A[1,1]*Yg)+c[1]
    # commensurability check
    for v in [(-A[0,0]*Lx,-A[1,0]*Lx),(-A[0,1]*Ly,-A[1,1]*Ly)]:
        n2=v[1]/(s3/2); n1=v[0]-0.5*n2
        assert abs(n2-round(n2))<1e-9 and abs(n1-round(n1))<1e-9, ('not commensurate',v)
    gb=(-A[0,0],-A[0,1],-A[1,0],-A[1,1]); Ar=dx_*dy_; R=np.roll
    def grads(f): return ((R(f,-1,0)-f)/dx_,(R(f,-1,1)-f)/dy_,(f-R(f,1,0))/dx_,(f-R(f,1,1))/dy_)
    def adj(a1,a2,b1,b2): return (R(a1,1,0)-a1)/dx_+(R(a2,1,1)-a2)/dy_+(b1-R(b1,-1,0))/dx_+(b2-R(b2,-1,1))/dy_
    def EG(z):
        tx,ty=z.reshape(2,Nx,Ny); Ux=grads(tx);Uy=grads(ty); E=0.; gu=[[],[]]
        for s in (0,1):
            exx=Ux[2*s]+gb[0]; eyy=Uy[2*s+1]+gb[3]; exy=0.5*(Ux[2*s+1]+gb[1]+Uy[2*s]+gb[2]); tr=exx+eyy
            E+=0.5*Ar*np.sum(mu*(exx**2+eyy**2+2*exy**2)+0.5*lam*tr**2)
            sxx=2*mu*exx+lam*tr; syy=2*mu*eyy+lam*tr; sxy=2*mu*exy; w=0.5*Ar
            gu[0]+=[w*sxx,w*sxy]; gu[1]+=[w*sxy,w*syy]
        G=np.stack([adj(*gu[0]),adj(*gu[1])])
        V,gx,gy=Vhex(tx+ubx,ty+uby,V0,V3); E+=Ar*np.sum(V); G[0]+=Ar*gx; G[1]+=Ar*gy
        G-=G.mean(axis=(1,2),keepdims=True)
        if sym: G=symm(G)
        return E,G.ravel()
    def mir_y(F):  # y -> -y : ux even, uy odd
        Fm=F[:,:,(-np.arange(Ny))%Ny]; return np.stack([Fm[0],-Fm[1]])
    def mir_x(F):
        Fm=F[:,(-np.arange(Nx))%Nx,:]; return np.stack([-Fm[0],Fm[1]])
    def symm(F):
        F=0.5*(F+mir_y(F)); return 0.5*(F+mir_x(F))
    t=time.time()
    r=minimize(EG,np.zeros(2*Nx*Ny) if x0 is None else x0,jac=True,method='L-BFGS-B',options=dict(maxiter=maxit,maxfun=2*maxit,gtol=1e-10,ftol=1e-15,maxcor=30))
    tx,ty=r.x.reshape(2,Nx,Ny)
    return dict(x=x,y=y,ux=tx+ubx,uy=ty+uby,E=r.fun,nit=r.nit,t=time.time()-t,xopt=r.x)
C2=np.array([1.,1/s3])   # centroid c2 (degenerate minimum when V3=sqrt3 V0)
def domain_field(ux,uy):
    """>0 closer to A-type stacking, <0 closer to c2-type"""
    return mismatch(ux-C2[0],uy-C2[1])-mismatch(ux,uy)
