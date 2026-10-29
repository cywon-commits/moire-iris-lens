import numpy as np, sys
from bubble import run
from bscan import measure
site,rho,H=sys.argv[1],float(sys.argv[2]),float(sys.argv[3])
xc,yc=(0.,0.) if site=='AA' else (0.,6.0)
R=run(H0=H,rho=rho,xc=xc,yc=yc); M=measure(R)
np.savez(f'bubsz_{site}_{rho}_{H}.npz',x=R['x'],ux=R['ux'],uy=R['uy'],h=R['h'],xc=xc,yc=yc)
print(f"{site} rho={rho} H0={H}: Hmax={M['Hmax']:.2f} r_edge={M['r_edge']:.1f} loops={[(l[0],l[1],l[2]) for l in M['loops']]}",flush=True)
