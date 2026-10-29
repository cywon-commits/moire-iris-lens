import numpy as np
from scipy.optimize import minimize
# a=1. fields ux,v,h on y-grid. wall on +x axis: Q4 (y<0): u=(-1,0) ; Q1 (y>0): u=(0,-1)
def solve(lam=1.,mu=1.,V0=0.05,kap=0.2,Kz=0.2,Ly=25.,n=1001,seed=1e-2,h0=None):
    y=np.linspace(-Ly,Ly,n); dy=y[1]-y[0]; Y=lam+2*mu
    w=np.sqrt(Y/V0)/(2*np.pi); wt=np.sqrt(mu/V0)/(2*np.pi)
    ux=-1+(2/np.pi)*np.arctan(np.exp(y/wt)); v=-(2/np.pi)*np.arctan(np.exp(y/w))
    h=seed*np.exp(-y**2/4) if h0 is None else h0
    def D(f): return np.diff(f)/dy
    def Dt(g): o=np.zeros(n); o[1:]+=g/dy; o[:-1]-=g/dy; return o
    def C(f): return (f[2:]-2*f[1:-1]+f[:-2])/dy**2
    def Ct(g): o=np.zeros(n); o[2:]+=g/dy**2; o[1:-1]-=2*g/dy**2; o[:-2]+=g/dy**2; return o
    fix=np.zeros(n,bool); fix[:3]=fix[-3:]=True
    def EG(z):
        ux,v,h=z.reshape(3,n)
        dh=D(h); e=D(v)+0.5*dh**2; g1=D(ux); c=C(h)
        p=2*np.pi
        E=dy*(np.sum(Y/2*e**2+mu/2*g1**2)+np.sum(kap/2*c**2)
              +np.sum(Kz/2*h**2+V0*(2-np.cos(p*ux)-np.cos(p*v))))
        gux=dy*(Dt(mu*g1)+V0*p*np.sin(p*ux))
        gv=dy*(Dt(Y*e)+V0*p*np.sin(p*v))
        gh=dy*(Dt(Y*e*dh)+Ct(kap*c)+Kz*h)
        G=np.stack([gux,gv,gh]); G[:,fix]=0
        return E,G.ravel()
    r=minimize(EG,np.concatenate([ux,v,h]),jac=True,method='L-BFGS-B',
               options=dict(maxiter=200000,maxfun=400000,gtol=1e-11,ftol=1e-15,maxcor=30))
    ux,v,h=r.x.reshape(3,n)
    absorbed=0.5*np.sum(np.diff(h)**2)/dy        # length taken by h (a=1)
    return dict(y=y,ux=ux,v=v,h=h,E=r.fun,f=absorbed,H=np.abs(h).max(),w=w,
                lb=(kap/Kz)**0.25,Lam=kap*Kz/(Y*V0),ok=r.success,nit=r.nit)
if __name__=="__main__":
    Y,V0=3.,0.05
    for lb in [0.5,1.,2.]:
        for Lam in [0.05,0.2,0.5,1.0,1.5]:
            P=np.sqrt(Lam*Y*V0); kap=P*lb**2; Kz=P/lb**2
            R=solve(V0=V0,kap=kap,Kz=Kz)
            print(f"lb={lb:4.1f} r=w/lb={R['w']/lb:4.2f} Lam={Lam:4.2f}  H={R['H']:.3f}  f={R['f']:.3f}  nit={R['nit']}")
