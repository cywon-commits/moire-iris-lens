import numpy as np, sys, json
from ten import run, nodes
res=[]
for sig in [2.5,3.5,5.0]:
    for p in [2.0,4.0]:
        R=run(p=p,sig=sig); nd=nodes(R,thr=7.0)
        D=max([np.hypot(a[0]-b[0],a[1]-b[1]) for a in nd for b in nd],default=0.)
        np.savez(f'tensig_{sig}_{p}.npz',x=R['x'],ux=R['ux'],uy=R['uy'])
        res.append(dict(sig=sig,p=p,D=D,nodes=nd)); print(f'sig={sig} p={p} nodes={len(nd)} D={D:.2f} pos={[(round(q[0],1),round(q[1],1)) for q in nd]}',flush=True)
json.dump(res,open('tensig.json','w'))
