import numpy as np, sys
from scipy.ndimage import maximum_filter
from relax2d import run
lam=float(sys.argv[1]); Y,V0=3.,0.05
R=run(L=16.,lam=lam,V0=V0,kap=1.,Kz=1.,mode='rect6',maxit=120000)
ux,uy=R['ux'],R['uy']; x=R['x']; m=np.hypot(ux-np.round(ux),uy-np.round(uy))
pk=(m==maximum_filter(m,size=9))&(m>0.65); pk[:8]=pk[-8:]=False; pk[:,:8]=pk[:,-8:]=False
I,J=np.nonzero(pk); nd=[(x[i],x[j]) for i,j in zip(I,J)]
D=max([np.hypot(a[0]-b[0],a[1]-b[1]) for a in nd for b in nd],default=0)
np.savez(f'rect6_{lam}.npz',x=x,ux=ux,uy=uy)
print(f'lam={lam} nodes={len(nd)} D={D:.2f} pos={[(round(a,1),round(b,1)) for a,b in nd]} nit={R["nit"]}')
