import sys,json,os,numpy as np
from pair2 import run
mode,d,alpha,V0=sys.argv[1],float(sys.argv[2]),float(sys.argv[3]),float(sys.argv[4])
defs={'pm':[(-d/2,0.,4.),(d/2,0.,-4.)],'Radd':[(d/2,0.,-4.)],'pp':[(-d/2,0.,4.),(d/2,0.,4.)]}[mode]
tag=f'opp_{mode}_d{d}_a{alpha:.0f}_V{V0}'
if os.path.exists(tag+'.json'): sys.exit()
R=run(defs,V0=V0,alpha=alpha,lam=-0.5)
if V0>0: np.savez(tag+'.npz',x=R['x'],ux=R['ux'],uy=R['uy'],core=R['core'])
json.dump(dict(E=R['E']),open(tag+'.json','w')); print(tag,R['E'],flush=True)
