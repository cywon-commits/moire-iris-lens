import numpy as np, scipy.linalg as sl
# dimensionless (lengths in l_b, kappa=Kz=1): Q[h]=∫ h''^2 - s(y) h'^2 + h^2,
# s(y)= (2/sqrt(Lam)) sech(y/r).  Threshold Lam_c(r): smallest generalized eigenvalue crosses 0.
def mineig(Lam,r,Ly=None,n=1600):
    Ly = Ly or max(30, 15*r)
    y=np.linspace(-Ly,Ly,n); dy=y[1]-y[0]
    ym=0.5*(y[1:]+y[:-1])
    s=2/np.sqrt(Lam)/np.cosh(ym/r)
    D1=(np.eye(n,k=1)-np.eye(n))[:-1]/dy           # links
    D2=(np.eye(n,k=1)-2*np.eye(n)+np.eye(n,k=-1))[1:-1]/dy**2
    H=(D2.T@D2)*dy - (D1.T@(s[:,None]*D1))*dy + np.eye(n)*dy
    H=H[2:-2,2:-2]                                  # clamp ends
    return sl.eigh(H,eigvals_only=True,subset_by_index=[0,0])[0]
def Lamc(r):
    lo,hi=1e-4,2.0
    for _ in range(40):
        m=np.sqrt(lo*hi)
        if mineig(m,r)<0: lo=m
        else: hi=m
    return np.sqrt(lo*hi)
if __name__=="__main__":
  for r in [0.05,0.1,0.2,0.4,0.8,1.6,3.2,6.4]:
      print(f"r={r:5.2f}  Lam_c={Lamc(r):.4f}  narrow pi^2 r^2/2={np.pi**2*r**2/2:.4f}")
