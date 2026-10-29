import numpy as np, json
from ridge1d import solve
from lin import mineig
Y,V0=3.,0.05
out={}
for lb in [0.5,1.0]:
    rows=[];h0=None
    for Lam in np.linspace(0.02,0.6,16):
        P=np.sqrt(Lam*Y*V0); R=solve(V0=V0,kap=P*lb**2,Kz=P/lb**2,h0=None if h0 is None else h0+1e-3*np.exp(-np.linspace(-25,25,1001)**2/4))
        h0=R['h']; rows.append([Lam,R['H'],R['f']])
        print(lb,round(Lam,3),round(R['H'],4),round(R['f'],4),flush=True)
        if abs(Lam-0.1)<0.02: np.savez(f'prof_lb{lb}.npz',**{k:R[k] for k in ['y','ux','v','h']})
    out[str(lb)]=rows
json.dump(out,open('scan.json','w'))
