import numpy as np, json
from invert import observe, fit, summarize
kap0=1j/30; rng=np.random.default_rng(5); res={}
z,hd,AA=observe('pair_d32.0_V0.005.npz'); truth=[(-16,0,4),(16,0,4)]
def err(out):
    if len(out)<2: return np.nan,np.nan
    e=[]; 
    for tx,ty,tc in truth:
        k=np.argmin([np.hypot(x-tx,y-ty) for x,y,c in out]); x,y,c=out[k]; e.append((np.hypot(x-tx,y-ty),abs(c-tc)/tc))
    e=np.array(e); return e[:,0].mean(),e[:,1].mean()
# (a) normal nodes only
sel=hd>0; b=fit(z[sel],AA[sel],2,kap0); out,rms,bic,p=summarize(b,2,sel.sum())
print('normal nodes only (',sel.sum(),'nodes):',out,' pos err %.2fa, C err %.1f%%'%(err(out)[0],100*err(out)[1]))
res['normal_only']=err(out)
# (c) position noise
for sig in [0.25,0.5,1.0,2.0]:
    E=[]
    for t in range(12):
        zn=z+sig*(rng.standard_normal(len(z))+1j*rng.standard_normal(len(z)))/np.sqrt(2)
        b=fit(zn,AA,2,kap0,ntry=15,rng=rng); out,rms,bic,p=summarize(b,2,len(z)); E.append(err(out))
    E=np.array(E); print(f'noise sigma={sig}a: mean position error {np.nanmean(E[:,0]):.2f}a, mean C error {100*np.nanmean(E[:,1]):.1f}%')
    res[f'noise_{sig}']=[float(np.nanmean(E[:,0])),float(np.nanmean(E[:,1]))]
# also normal-only with noise 1a
E=[]
for t in range(12):
    zn=z[sel]+1.0*(rng.standard_normal(sel.sum())+1j*rng.standard_normal(sel.sum()))/np.sqrt(2)
    b=fit(zn,AA[sel],2,kap0,ntry=15,rng=rng); out,rms,bic,p=summarize(b,2,sel.sum()); E.append(err(out))
E=np.array(E); print(f'normal-only + noise 1a: pos err {np.nanmean(E[:,0]):.2f}a, C err {100*np.nanmean(E[:,1]):.1f}%')
res['normal_noise1']=[float(np.nanmean(E[:,0])),float(np.nanmean(E[:,1]))]
json.dump(res,open('invert_tests.json','w'))
