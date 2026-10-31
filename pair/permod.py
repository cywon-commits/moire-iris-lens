# moire elastic constants from periodic cells. registry u = c1 + P z + Q zbar + phi(periodic)
# moire displacement gradient: P = kap(1-Ua), Q = -kap Ub  (u = kap(z-U), U = Ua z + Ub zbar)
import sys,json,time,numpy as np
sys.path.insert(0,'/home/claude/ridge/hex'); from hexcore import Vhex
from scipy.optimize import minimize
s3=np.sqrt(3.); c1=np.array([0.5,1/(2*s3)])
def cell(alpha,th,nx=2,ny=2):
    if alpha==90: return nx*s3/th, ny/th, nx*1j*s3, -ny+0j      # kap*Lx = mx, kap*i*Ly = my
    return nx/th, ny*s3/th, nx+0j, ny*1j*s3
def solve(alpha,ed,gam,th=1/30.,V0=0.005,lam=-0.5,mu=1.,nx=2,ny=2,Nx=None,Ny=None,x0=None,seed=0):
    kap=th*np.exp(1j*np.radians(alpha)); L0x,L0y,mx,my=cell(alpha,th,nx,ny)
    Lx=L0x*(1+ed+gam); Ly=L0y*(1+ed-gam)
    P=0.5*(mx/Lx+my/(1j*Ly)); Q=0.5*(mx/Lx-my/(1j*Ly))
    Ua=1-P/kap; Ub=-Q/kap
    Nx=Nx or int(round(L0x/0.45)); Ny=Ny or int(round(L0y/0.45))
    dx=Lx/Nx; dy=Ly/Ny; x=np.arange(Nx)*dx; y=np.arange(Ny)*dy; X,Y=np.meshgrid(x,y,indexing='ij'); Z=X+1j*Y
    W=P*Z+Q*np.conj(Z); bx=c1[0]+W.real; by=c1[1]+W.imag
    g=(P+Q, 1j*(P-Q))
    gb=(g[0].real,g[1].real,g[0].imag,g[1].imag)
    A=dx*dy; R=np.roll; V3=s3*V0
    def grads(f): return ((R(f,-1,0)-f)/dx,(R(f,-1,1)-f)/dy,(f-R(f,1,0))/dx,(f-R(f,1,1))/dy)
    def adj(a1,a2,b1,b2): return (R(a1,1,0)-a1)/dx+(R(a2,1,1)-a2)/dy+(b1-R(b1,-1,0))/dx+(b2-R(b2,-1,1))/dy
    def EG(z,parts=False):
        fx,fy=z.reshape(2,Nx,Ny); Ux=grads(fx);Uy=grads(fy); gu=[[],[]]; Eel=0.
        for s in (0,1):
            exx=Ux[2*s]+gb[0]; eyy=Uy[2*s+1]+gb[3]; exy=0.5*(Ux[2*s+1]+gb[1]+Uy[2*s]+gb[2]); tr=exx+eyy
            Eel+=0.5*A*np.sum(mu*(exx**2+eyy**2+2*exy**2)+0.5*lam*tr**2)
            w=0.5*A; gu[0]+=[w*(2*mu*exx+lam*tr),w*2*mu*exy]; gu[1]+=[w*2*mu*exy,w*(2*mu*eyy+lam*tr)]
        G=np.stack([adj(*gu[0]),adj(*gu[1])])
        V,gx,gy=Vhex(bx+fx,by+fy,V0,V3); EV=A*np.sum(V); G[0]+=A*gx; G[1]+=A*gy
        G-=G.mean(axis=(1,2),keepdims=True)
        if parts: return Eel,EV
        return Eel+EV,G.ravel()
    if x0 is None:
        rng=np.random.default_rng(seed); x0=1e-3*rng.standard_normal(2*Nx*Ny)
    t=time.time()
    r=minimize(EG,x0,jac=True,method='L-BFGS-B',options=dict(maxiter=50000,maxfun=100000,gtol=1e-11,ftol=1e-16,maxcor=30))
    Eel,EV=EG(r.x,True); Ar=Lx*Ly
    return dict(alpha=alpha,ed=ed,gam=gam,Ua=[Ua.real,Ua.imag],Ub=[Ub.real,Ub.imag],e=float(r.fun/Ar),eel=float(Eel/Ar),eV=float(EV/Ar),
                Lx=Lx,Ly=Ly,nit=int(r.nit),t=time.time()-t,x=r.x)
if __name__=="__main__":
    a=float(sys.argv[1]); ed=float(sys.argv[2]); gam=float(sys.argv[3])
    R=solve(a,ed,gam); x=R.pop('x')
    print(json.dumps(R),flush=True)
