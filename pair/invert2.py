import numpy as np, json
from scipy.optimize import least_squares
from invert import observe
def model(p,z,K,kap):
    u=p[0]+1j*p[1]+kap*z
    for k in range(K): u=u-p[4+3*k]/np.conj(z-(p[2+3*k]+1j*p[3+3*k]))
    return u
def fitk(z,AA,K,kap,rng,ntry=60):
    best=None
    for t in range(ntry):
        p=[(AA-kap*z).mean().real,(AA-kap*z).mean().imag]
        for k in range(K): p+=[rng.uniform(-40,40),rng.uniform(-40,40),rng.uniform(1,8)]
        lo=[-np.inf,-np.inf]+[-70,-70,0.]*K; hi=[np.inf,np.inf]+[70,70,30.]*K
        try: r=least_squares(lambda q: np.concatenate([(model(q,z,K,kap)-AA).real,(model(q,z,K,kap)-AA).imag]),p,bounds=(lo,hi),method='trf',max_nfev=3000)
        except ValueError: continue
        if best is None or r.cost<best.cost: best=r
    p=best.x; return [(p[2+3*k],p[3+3*k],p[4+3*k]) for k in range(K)]
truth=[(-16,0,4),(16,0,4)]
def err(out):
    e=[]
    for tx,ty,tc in truth:
        k=np.argmin([np.hypot(x-tx,y-ty) for x,y,c in out]); x,y,c=out[k]; e.append((np.hypot(x-tx,y-ty),abs(c-tc)/tc))
    return np.array(e).mean(0)
rng=np.random.default_rng(3); kap=1j/30
z,hd,AA=observe('pair_d32.0_V0.005.npz'); res={}
for name,sel in [('all nodes',np.ones(len(z),bool)),('normal only',hd>0)]:
    for sig in [0.0,0.5,1.0,2.0,3.0]:
        E=[]
        for t in range(1 if sig==0 else 10):
            zn=z[sel]+sig*(rng.standard_normal(sel.sum())+1j*rng.standard_normal(sel.sum()))/np.sqrt(2)
            E.append(err(fitk(zn,AA[sel],2,kap,rng,ntry=30)))
        E=np.array(E); res[f'{name}_{sig}']=E.mean(0).tolist()
        print(f'{name:12s} ({sel.sum()} nodes) noise {sig}a: position error {E[:,0].mean():.2f}a (median {np.median(E[:,0]):.2f}), C error {100*E[:,1].mean():.1f}%',flush=True)
json.dump(res,open('invert2.json','w'))
