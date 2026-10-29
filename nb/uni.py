import numpy as np, sys
import userfield, relaxnb
from anal import loopdata
case=sys.argv[1]; hl=float(sys.argv[2])
x0,y0,k,Ds,Cf=userfield.CASES[case]
userfield.CASES[case+'u']=(x0,y0,k,1e-9,Cf); relaxnb.CASES[case+'u']=(x0,y0,k,1e-9,Cf)
R=relaxnb.run(case+'u',hole=hl)
f=f'nb_{case}u_h{hl}.npz'
np.savez(f,**{kk:R[kk] for kk in ['x','ux','uy','ux0','uy0','pore']},X0=R['X0'],Y0=R['Y0'],k=R['k'])
print(case,'uniform start, pore',hl,'E=',round(R['E'],3),' loop (walls,Q) at +2a:',loopdata(f,rad=hl+2)[1:],flush=True)
