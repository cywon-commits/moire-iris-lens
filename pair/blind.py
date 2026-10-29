import numpy as np, json
from pair import run
truth=[(-20.,14.,2.5),(9.,-6.,6.0),(27.,22.,3.5)]
R=run(truth,V0=0.005)
np.savez('blind.npz',x=R['x'],ux=R['ux'],uy=R['uy'],core=R['core'])
json.dump(dict(truth=truth,E=R['E']),open('blind.json','w')); print('blind done',R['E'],R['nit'],round(R['t']))
