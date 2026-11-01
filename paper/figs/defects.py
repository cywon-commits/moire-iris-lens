import numpy as np
from scipy.spatial import cKDTree
from lat import *
s3=np.sqrt(3); Acell=s3/2; Aat=Acell/2
def bonds(X,cut=0.66):
    T=cKDTree(X); return np.array(sorted(T.query_pairs(cut)))
def radial(z,C):
    # area-preserving radial map with far field -C/conj(z): r'^2 = r^2 - 2C
    r=np.abs(z); rn=np.sqrt(np.maximum(r**2-2*C,0)); return z/np.maximum(r,1e-12)*rn
def vacancy(X,rh=4.2,close=0.7,c0=0j):
    z=X[:,0]+1j*X[:,1]-c0; k=np.abs(z)>rh; C=close*rh**2/2           # removed area pi rh^2; far charge C (partially closed iris)
    zn=radial(z[k],C)+c0; tag=np.zeros(k.sum(),int); tag[np.abs(z[k])<rh+1.2]=1
    return np.c_[zn.real,zn.imag],C,tag
def island(X,rh=4.0,Cabs=4.0,c0=0j,twist=np.radians(9)):
    z=X[:,0]+1j*X[:,1]-c0; k=np.abs(z)>rh
    zo=radial(z[k],-Cabs)+c0                                        # outer lattice pushed out
    Rin=np.sqrt(rh**2+2*Cabs); I=rot(honeycomb(Rin+2),twist); zi=I[:,0]+1j*I[:,1]; zi=zi[np.abs(zi)<Rin-0.25]+c0
    Z=np.r_[zi,zo]; tag=np.r_[np.full(len(zi),2),np.zeros(len(zo),int)]
    return np.c_[Z.real,Z.imag],-Cabs,tag
def rotpatch(X,rh=5.0,phi=np.radians(11)):
    z=X[:,0]+1j*X[:,1]; r=np.abs(z); c=phi*rh**2
    ang=np.where(r<rh,phi,phi*rh**2/np.maximum(r,1e-9)**2)          # rigid inside, tangential ~ 1/r outside
    zn=z*np.exp(1j*ang); tag=(r<rh).astype(int)*2
    return np.c_[zn.real,zn.imag],-1j*c,tag                          # C = -i c  (delta u = -C/zbar = i c z/|z|^2, ccw)
def neutral(X,sep=10.0):
    Y,C1,t1=vacancy(X,rh=4.0,close=0.7,c0=-sep)
    Z,C2,t2=island(Y,rh=0.01,Cabs=C1,c0=sep+0j)
    t=np.r_[t2[t2==2],t1[-(t2==0).sum():]] if False else None
    # recompute tags
    zz=Z[:,0]+1j*Z[:,1]; tag=np.zeros(len(Z),int); tag[np.abs(zz-sep)<np.sqrt(2*C1)-0.2]=2; tag[(np.abs(zz+sep)<4.0)]=1
    return Z,C1+C2,tag
def edge(X,b=1.0,nu=0.2,core=0.7):
    x=X[:,0]; y=X[:,1]+1e-9; r2=x*x+y*y+core**2; t=np.arctan2(y,x)
    ux=b/(2*np.pi)*(t+x*y/(2*(1-nu)*r2)); uy=-b/(2*np.pi)*((1-2*nu)/(4*(1-nu))*np.log(r2)+(x*x-y*y)/(4*(1-nu)*r2))
    tag=((x*x+y*y)<4).astype(int)
    return np.c_[x+ux,X[:,1]+uy],0,tag
