import os
import numpy as np, sys, json
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','hex'))
from scipy.ndimage import map_coordinates
from matplotlib.path import Path
from affine import domain_field
from bubble import run, c1, s3
def measure(R,dg=0.5,rmax=20.):
    x=R['x']; hgrid=x[1]-x[0]; hh=R['h']; g=np.exp(-(np.maximum(hh,0)/dg)**2)
    X,Y=np.meshgrid(x,x,indexing='ij'); rr=np.hypot(X-R['xc'],Y-R['yc'])
    deln=(g<0.5)&(rr<rmax); r_edge=rr[deln].max() if deln.any() else 0.
    gx,gy=np.gradient(hh,hgrid); dA=0.5*np.sum(gx**2+gy**2)*hgrid**2
    f_=domain_field(R['ux'],R['uy']); a1=np.array([1.,0]); a2=np.array([.5,s3/2])
    res=[]
    for m in (0.5,1.0,2.0,3.0):
        rl=r_edge+m; th=np.linspace(0,2*np.pi,4001)[:-1]
        idx=[(R['xc']+rl*np.cos(th)-x[0])/hgrid,(R['yc']+rl*np.sin(th)-x[0])/hgrid]
        ux=map_coordinates(R['ux'],idx,order=1); uy=map_coordinates(R['uy'],idx,order=1)
        v=map_coordinates(f_,idx,order=1); s=np.sign(v); walls=int((s!=np.roll(s,-1)).sum())
        P=Path(np.c_[ux,uy]); Q=sum(P.contains_point(c1+n1*a1+n2*a2) for n1 in range(-8,9) for n2 in range(-8,9))
        px=R['xc']+rl*np.cos(th); py=R['yc']+rl*np.sin(th)
        sc=np.sqrt(3)/(4*np.pi); lam_s=12./(4*np.pi/np.sqrt(3))
        mx=c1[0]-sc*py/lam_s; my=c1[1]+sc*px/lam_s
        dr=np.mean((ux-mx)*np.cos(th)+(uy-my)*np.sin(th))
        res.append((round(rl,2),walls,int(Q),round(dA/(2*np.pi*rl),3),round(-dr,3)))
    return dict(r_edge=float(r_edge),dA=float(dA),Hmax=float(hh.max()),loops=res)
if __name__=="__main__":
    site=sys.argv[1]; Hs=[float(v) for v in sys.argv[2].split(',')]
    xc,yc=(0.,0.) if site=='AA' else (0.,6.0)
    out=[]
    for H in Hs:
        R=run(H0=H,xc=xc,yc=yc); M=measure(R); M.update(H0=H,E=float(R['E']),nit=int(R['nit']))
        np.savez(f'bub_{site}_{H}.npz',x=R['x'],ux=R['ux'],uy=R['uy'],h=R['h'],xc=xc,yc=yc)
        out.append(M)
        print(f"{site} H0={H}: Hmax={M['Hmax']:.2f} r_edge={M['r_edge']:.1f} dA={M['dA']:.1f}  loops (r, walls, Q, delta_pred)={M['loops']}  nit={R['nit']} {R['t']:.0f}s",flush=True)
    json.dump(out,open(f'bscan_{site}_{sys.argv[2]}.json','w'))
