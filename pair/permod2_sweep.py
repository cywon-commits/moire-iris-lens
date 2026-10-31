import json,sys,itertools
from permod2 import solve
a=float(sys.argv[1]); out=[]
pts=[(0,0)]
for v in (-.03,-.02,-.01,.01,.02,.03):
    pts+=[(v,0),(1j*v,0),(0,v),(0,1j*v)]
pts+=[(.01+.01j,0),(.01-.01j,0),(-.01+.01j,0),(.01,.01),(.01,.01j),(.01j,.01),(.01j,.01j),(0,.01+.01j),(.02+.02j,0),(-.02+.02j,0)]
for Ua,Ub in pts:
    R=solve(a,Ua,Ub); R.pop('x'); R.pop('EG'); out.append(R); print(Ua,Ub,R['e'],R['nit'],round(R['t']),flush=True)
    json.dump(out,open(f'permod2_a{a:.0f}.json','w'))
