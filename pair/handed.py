import numpy as np, json, glob
def omega_ring(f,xd,r0=4.,r1=12.):
    D=np.load(f); x=D['x']; h=x[1]-x[0]; ux,uy=D['ux'],D['uy']
    duy_dx=np.gradient(uy,h,axis=0); dux_dy=np.gradient(ux,h,axis=1); w=0.5*(duy_dx-dux_dy)
    X,Y=np.meshgrid(x,x,indexing='ij'); R=np.hypot(X-xd,Y); m=(R>r0)&(R<r1)&(~D['core'])
    return float(w[m].mean())*1e3
E={}
for f in glob.glob('*.json'):
    try: d=json.load(open(f))
    except Exception: continue
    if 'mode' in d: E[(d['mode'],d['d'],d['V0'],d.get('alpha',90.),d.get('orient','x'),d.get('sym',False),d.get('symm'))]=d['E']
ds=[8.,16.,26.,40.,56.]; R=np.sqrt(120.)
Ue={d:E[('pair',d,0.0,90.,'x',False,None)]-2*E[('R',d,0.0,90.,'x',False,None)] for d in ds}
bg=min(E[('none',0.0,0.005,0.,'x',False,None)],E[('none',0.0,0.005,0.,'x',True,None)])
rows=[]
for d in ds:
    single=min(E[('R',d,0.005,0.,'x',False,None)],E[('R',d,0.005,0.,'x',True,None)])
    br={'free':E.get(('pair',d,0.005,0.,'x',False,None)),'achiral':E.get(('pair',d,0.005,0.,'x',True,None)),
        'opposite (xm)':E.get(('pair',d,0.005,0.,'x',False,'xm')),'same (c2)':E.get(('pair',d,0.005,0.,'x',False,'c2'))}
    U={k:(v-2*single+bg-Ue[d] if v is not None else None) for k,v in br.items()}
    Ua=E[('pair',d,0.005,0.,'x',True,None)]-2*E[('R',d,0.005,0.,'x',True,None)]+E[('none',0.0,0.005,0.,'x',True,None)]-Ue[d]
    hand={}
    for k,f in [('free',f'pair_d{d}_V0.005_a0_x.npz'),('opposite (xm)',f'pair_d{d}_V0.005_a0_x_xm.npz'),('same (c2)',f'pair_d{d}_V0.005_a0_x_c2.npz')]:
        try: hand[k]=(round(omega_ring(f,-d/2),2),round(omega_ring(f,d/2),2))
        except Exception: pass
    best=min((v,k) for k,v in U.items() if v is not None)
    rows.append(dict(d=d,U=U,U_lens_only=Ua,hand=hand,best=best[1]))
    print(f"d={d:4.0f} (d/R={d/R:.2f}) lens-only {Ua:+.2f} | "+'  '.join(f"{k}: {v:+.2f}" for k,v in U.items() if v is not None)+f"  -> ground: {best[1]} {best[0]:+.2f}")
    print('       halo rotation (1e-3) left,right:',hand)
sing={}
for d in ds: sing[d]=round(omega_ring(f'R_d{d}_V0.005_a0_x.npz',d/2),2)
print('single defect halo rotation (1e-3):',sing)
json.dump(rows,open('handed.json','w'))
