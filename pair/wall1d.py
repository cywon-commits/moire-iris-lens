import os
import numpy as np, sys
from scipy.optimize import minimize
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','hex')); from hexcore import Vhex
s3=np.sqrt(3); V0=0.005; b=np.array([0.,1/s3])
def gamma(chi,lam,mu=1.,L=40.,N=801):
    # chi = angle between Burgers vector b and wall normal n  (0 = tensile, 90 = shear)
    n=np.array([np.sin(chi),np.cos(chi)])   # b along y -> b.n = cos chi
    s=np.linspace(-L,L,N); h=s[1]-s[0]
    u0=np.outer(0.5*(1+np.tanh(s/2)),b)
    def E(z):
        u=z.reshape(N,2); up=np.diff(u,axis=0)/h
        un=up@n; e=0.5*(np.einsum('ki,j->kij',up,n)+np.einsum('i,kj->kij',n,up))
        f=mu*np.sum(e**2,axis=(1,2))+0.5*lam*(un)**2
        um=0.5*(u[1:]+u[:-1]); V,_,_=Vhex(um[:,0],um[:,1],V0,s3*V0)
        return h*np.sum(f+V)
    z0=u0.ravel(); fixed=np.zeros((N,2),bool); fixed[0]=fixed[-1]=True
    def f(z):
        zz=z0.copy(); zz[~fixed.ravel()]=z; return E(zz)
    r=minimize(f,z0[~fixed.ravel()],method='L-BFGS-B',options=dict(maxiter=20000))
    return r.fun
if __name__=='__main__':
    for lam in [1.,-0.5]:
        g=[gamma(np.radians(c),lam) for c in (0,30,60,90)]
        print(f'lam={lam}: gamma(chi=0 tensile,30,60,90 shear)=',np.round(g,4),' ratio t/s=',round(g[0]/g[3],3))
