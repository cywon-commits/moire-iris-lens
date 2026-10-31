import json,glob,os,numpy as np
from nodedisp import node_positions
s3=np.sqrt(3); n=np.arange(-12,13); N1,N2=np.meshgrid(n,n,indexing='ij'); Lv=((N1+0.5*N2)+1j*(s3/2*N2)).ravel()
res={}
for f in sorted(glob.glob('st_a*_L*.json')):
    J=json.load(open(f)); a=int(J['alpha']); Lp=J['Lp']
    if J['mode'] not in ('d','0'): continue
    eta=J['amp'] if J['mode']=='d' else 0.0
    zi=Lv/(np.exp(1j*np.radians(a))/30); m=np.abs(zi)<Lp-18; zi=zi[m]
    zz,r_=node_positions(f[:-5],a,rmax=Lp-18); 
    ok=r_<1e-8; d=(zz-zi)[ok]; z0=zi[ok]; r=np.abs(z0); k=r>1
    eb=np.sum((d[k]*np.conj(z0[k])).real)/np.sum(r[k]**2)       # best uniform dilation
    rot=np.sum((d[k]*np.conj(z0[k])).imag)/np.sum(r[k]**2)
    resid=np.sqrt(np.mean(np.abs(d[k]-eb*z0[k])**2))
    ec=eta/(1-eta); Delta=(eb-ec)*Lp
    res.setdefault(f'{a}_{Lp:.0f}',[]).append(dict(eta=eta,eps_b=eb,rot=rot,resid=resid,Delta=Delta,E=J['E'],nodes=int(k.sum())))
for key in res:
    res[key].sort(key=lambda x:x['eta']); print(key)
    for x in res[key]: print(f"  eta={x['eta']:+.3f} eps_b={x['eps_b']:+.4f} rot={x['rot']:+.4f} resid={x['resid']:.3f} Delta={x['Delta']:+.3f} E={x['E']:.4f} n={x['nodes']}")
json.dump(res,open('eta_scan.json','w'),indent=1)
