import numpy as np, sys
from bubble2 import run
tag=sys.argv[1]; kw=eval('dict('+sys.argv[2]+')')
R=run(**kw)
np.savez(f'wrk_{tag}.npz',x=R['x'],ux=R['ux'],uy=R['uy'],h=R['h'],xc=R['xc'],yc=R['yc'],**{k:v for k,v in kw.items() if k not in ('xc','yc')})
print(tag,kw,'E',round(R['E'],3),'hmax',round(R['h'].max(),2),'nit',R['nit'],round(R['t']),'s',flush=True)
