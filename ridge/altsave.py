import numpy as np, json, base64
from multilayer import run
L=16.; eps=1j/L; ts=np.linspace(0,1,9); x0=None; data={'t':[],'E':[],'m12':[],'m23':[]}
for t in ts:
    delta=t*(L/2)*(1+1j)
    R=run([0,eps,0],[0,0,-eps*delta],L=L,x0=x0); x0=R['xopt']
    U=np.array(R['U']); N=U.shape[-1]
    def mis(u): return np.hypot(*(u-np.round(u)))
    m12=np.roll(mis(U[1]-U[0]),(N//2,N//2),(0,1)); m23=np.roll(mis(U[2]-U[1]),(N//2,N//2),(0,1))
    enc=lambda m: base64.b64encode(np.clip(m/0.7071*255,0,255).astype(np.uint8).T.copy().tobytes()).decode()
    data['t'].append(float(t)); data['E'].append(float(R['E'])); data['m12'].append(enc(m12)); data['m23'].append(enc(m23))
    print(t,R['E'],flush=True)
data['N']=int(N)
json.dump(data,open('altviz.json','w'))
