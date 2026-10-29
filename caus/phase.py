import numpy as np, json
from quartic import count_grid
res={}
for phd in range(0,181,15):
    row=[]
    for aa in [0.1,0.2,0.35,0.5,0.7,0.9]:
        b,N,D=count_grid(aa,np.exp(1j*np.radians(phd)),L=3.2,n=81)
        da=(b[1]-b[0])**2; row.append(dict(a=aa,A4=float((N==4).sum()*da),A0=float((N==0).sum()*da)))
    res[phd]=row
    print(f'phi_m={phd:4d}: 4-image area ' + ' '.join(f"{r['A4']:6.2f}" for r in row) + '   | 0-image area ' + ' '.join(f"{r['A0']:5.2f}" for r in row),flush=True)
json.dump(res,open('phase.json','w'))
