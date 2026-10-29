import numpy as np, sys
from affine import run
from hexcore import s3
N=32; s=np.sqrt(3)/2/N*2; A=[[s,0],[0,s]]; c1=(0.5,1/(2*s3))
lam=float(sys.argv[1])
Rs=run(A,c=c1,sym=True,lam=lam)
rng=np.random.default_rng(0); x0=Rs['xopt']+1e-3*rng.standard_normal(Rs['xopt'].size)
Rf=run(A,c=c1,sym=False,lam=lam,x0=x0)
np.savez(f'chiral_lam{lam}.npz',x=Rs['x'],y=Rs['y'],uxs=Rs['ux'],uys=Rs['uy'],uxf=Rf['ux'],uyf=Rf['uy'],Es=Rs['E'],Ef=Rf['E'])
print(f'lam={lam} Y/mu={lam+2:.0f}  E_sym={Rs["E"]:.4f}  E_free={Rf["E"]:.4f}  dE={Rf["E"]-Rs["E"]:+.4f}  nit {Rs["nit"]},{Rf["nit"]}',flush=True)
