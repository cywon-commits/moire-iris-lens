import numpy as np, json, sys
from multilayer import run
L=16.; th=1/L; eps=1j*th          # pure twist, one moire cell per L x L
mode=sys.argv[1]
ts=np.linspace(0,1,9); out=[]; x0=None
for t in ts:
    delta=t*(L/2)*(1+1j)
    if mode=='hel': e=[0,eps,2*eps]; c=[0,0,eps*delta]
    else:           e=[0,eps,0];     c=[0,0,-eps*delta]
    R=run(e,c,L=L,maxit=60000,x0=x0); x0=R['xopt']
    out.append([t,R['E']]); print(mode,round(t,3),R['E'],R['nit'],flush=True)
    if t in (0,1): np.savez(f'tri_{mode}_{int(t)}.npz',x=R['x'],U=np.array(R['U']))
json.dump(out,open(f'scan3_{mode}.json','w'))
