import numpy as np
from quartic import count_grid
def has4(a,phd,n=141,L=3.0):
    b,N,D=count_grid(a,np.exp(1j*np.radians(phd)),L=L,n=n); return (N==4).sum()>0
for phd in [15,30,60,90,120,150]:
    lo,hi=0.0,0.99
    for _ in range(9):
        mid=0.5*(lo+hi)
        if has4(mid,phd): hi=mid
        else: lo=mid
    print(f'phi_m={phd:3d}: a_c={0.5*(lo+hi):.3f}   sin(phi/2)={np.sin(np.radians(phd)/2):.3f}   sin(phi)={np.sin(np.radians(phd)):.3f}   tan(phi/4)={np.tan(np.radians(phd)/4):.3f}',flush=True)
