"""Check wall valence z vs AA winding w on circles around every core: z >= 2 L_min(|w|), z even. Output: zw.json."""
import json,numpy as np
from topo import circle,frac,wind_points,zcount
def Lmin(w):
    if w==0: return 0
    best=99
    for a in range(1,12):
        for b in range(1,12):
            for c in range(1,24):
                for c0 in range(0,a+b):
                    if sum(1 for i in range(a) for j in range(b) if c0<=i+j<c0+c)>=w: best=min(best,a+b+c)
    return best
R=json.load(open('topo_all.json')); out=[]
for r in R:
    D=np.load(r['file']+'.npz')
    for p in r['cores']:
        for rad in (5.,7.,9.):
            out.append(dict(fam=r['fam'],r=rad,w=wind_points(np.stack(frac(*circle(D,p['core'][0],p['core'][1],rad)),-1)),z=zcount(D,p['core'][0],p['core'][1],rad)))
T={w:Lmin(w) for w in range(0,10)}
viol=[o for o in out if o['z']<2*T.get(abs(o['w']),99) or o['z']%2]
print('L_min',T,' circles',len(out),' violations',len(viol))
json.dump(out,open('zw.json','w'))
