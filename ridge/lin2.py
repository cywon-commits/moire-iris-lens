from lin import mineig
import numpy as np
def Lamc(r,n,Ly):
    lo,hi=1e-5,2.0
    for _ in range(35):
        m=np.sqrt(lo*hi)
        if mineig(m,r,Ly,n)<0: lo=m
        else: hi=m
    return np.sqrt(lo*hi)
for r in [0.02,0.05,0.1]:
    for n in [1600,3200]:
        print(r,n,Lamc(r,n,12),np.pi**2*r**2/2)
