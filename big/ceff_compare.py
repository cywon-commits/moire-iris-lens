"""Compare measured loop charge dQ(rho) with the dressed-charge prediction -(2pi/Ac)Re(conj(kappa) C_eff(rho))."""
import json,numpy as np
from ceff import ceff
th=1/30.; Ac=np.sqrt(3)/2
R=json.load(open('big_res.json')); out={}
for name,f,a in [('tw pull d40','B_tw_pull40',90),('tw push d40','B_tw_push40',90)]+[('mm pull d26','B_mm_pull26',0),('mm push d26','B_mm_push26',0)]:
    if name not in R: continue
    bg='B_bg_tw' if a==90 else 'B_bg_mm'
    rs=np.array(R[name]['r']); c=ceff(f,a,rs,ref=bg); k=th*np.exp(1j*np.radians(a))
    pred=-2*np.pi/Ac*(np.conj(k)*c).real
    dq=np.array(R[name]['dQ']); da=np.array(R[name]['dQarea']); dl=np.array(R[name]['dQlin'])
    out[name]=dict(r=rs.tolist(),ReC=c.real.tolist(),ImC=c.imag.tolist(),pred=pred.tolist())
    print(name)
    for a_,b_ in [(20,50),(50,80),(80,110),(110,140)]:
        m=(rs>=a_)&(rs<b_)
        print(f'  [{a_:3d}-{b_:3d}] measured dQ {dq[m].mean():+.2f}  dQarea {da[m].mean():+.2f}  | dressed pred {pred[m].mean():+.2f}  (ReC {c.real[m].mean():+.2f}, ImC {c.imag[m].mean():+.2f}) | bare-C lin {dl[m].mean():+.2f}')
json.dump(out,open('ceff_compare.json','w'))
