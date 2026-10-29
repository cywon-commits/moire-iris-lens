import numpy as np, json, sys
from multilayer import run
L=16.; eps=(1+1j)/L; Y,V0=3.,0.05; lb=float(sys.argv[2]); d1=0.3
Lams=[float(v) for v in sys.argv[1].split(',')]
out=[]
for Lam in Lams:
    P=np.sqrt(Lam*Y*V0)
    R=run([0,eps,0],L=L,V0=V0,kap=P*lb**2,Kz=P/lb**2,d1=d1,zfree=True,Kext=P/lb**2,maxit=80000,hseed=1e-2)
    H=np.array(R['H']); H-=H.mean(axis=(1,2),keepdims=True)
    m2=np.abs(H[1]).max(); asym=np.abs(H[0]+H[2]).max(); breath=np.abs(H[2]-H[0]).max()
    out.append([Lam,m2,asym,breath]); print(Lam,'h2 max',round(m2,4),'|h1+h3|',round(asym,4),'breathing',round(breath,4),R['nit'],flush=True)
    np.savez(f'sigz_{lb}_{Lam}.npz',x=R['x'],H=H,U=np.array(R['U']))
json.dump(out,open(f'sigz_{lb}_{sys.argv[1]}.json','w'))
