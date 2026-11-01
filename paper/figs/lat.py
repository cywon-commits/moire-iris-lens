import numpy as np
s3=np.sqrt(3)
a1=np.array([1.,0.]); a2=np.array([0.5,s3/2])
def honeycomb(L):
    n=int(L*1.6)+2; I,J=np.meshgrid(np.arange(-n,n+1),np.arange(-n,n+1),indexing='ij')
    P=I[...,None]*a1+J[...,None]*a2; P=P.reshape(-1,2)
    A=P; Bs=P+np.array([0.5,s3/6])          # two sublattices, bond length 1/sqrt3
    X=np.vstack([A,Bs]); X-=np.array([0.25,s3/12])
    k=(np.abs(X[:,0])<L)&(np.abs(X[:,1])<L); return X[k]
def rot(X,t):
    c,s=np.cos(t),np.sin(t); return X@np.array([[c,s],[-s,c]])
def apply(X,f):
    z=X[:,0]+1j*X[:,1]; d=f(z); return np.c_[X[:,0]+d.real,X[:,1]+d.imag]
