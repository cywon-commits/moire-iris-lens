import os
import numpy as np, sys, json
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','hex'))
from iris_an import nodes, node_handedness
theta=1/40
def images(beta,R2):
    """lens: beta = z + i R2/conj(z). returns list of (z, parity)"""
    b=abs(beta); 
    if R2==0: return [(beta,1)]
    disc=b**4-4*R2**2
    if disc<0: return []
    out=[]
    for sgn,par in [(1,1),(-1,-1)]:
        r=np.sqrt((b**2+sgn*np.sqrt(disc))/2); phi=np.angle(beta)-np.arctan(R2/r**2); out.append((r*np.exp(1j*phi),par))
    return out
def analyse(site,C,Rmax=55):
    D=np.load(f'lens_{site}_C{C}.npz'); x=D['x']; o=D['u0']
    c1=np.array([0.5,1/(2*np.sqrt(3))]); a1=np.array([1.,0]); a2=np.array([.5,np.sqrt(3)/2])
    B=[]
    for n1 in range(-14,15):
        for n2 in range(-14,15):
            q=c1+n1*a1+n2*a2-o; rho=q[0]+1j*q[1]
            if 1e-9<abs(rho)<12: B.append(rho/(1j*theta))
    B=np.array(B)
    P=nodes(x,D['ux'],D['uy'],D['core']); P=P[np.hypot(P[:,0],P[:,1])<Rmax]; hd=node_handedness(x,D['ux'],D['uy'],P)
    Z=P[:,0]+1j*P[:,1]; R2=C/theta
    pred=[(b,)+im for b in B for im in images(b,R2)]
    rows=[]
    for z,par in zip(Z,hd):
        cand=[p for p in pred if p[2]==par]
        if not cand: continue
        k=int(np.argmin([abs(z-p[1]) for p in cand])); b,zp,_=cand[k]
        rows.append(dict(beta=abs(b),r=abs(z),rpred=abs(zp),dphi=float(np.degrees(np.angle(z/b))),dphipred=float(np.degrees(np.angle(zp/b))),par=int(par),err=abs(z-zp)))
    swallowed=[abs(b) for b in B if abs(b)<np.sqrt(2*R2) and abs(b)>0]
    return rows,swallowed,B,Z,hd
if __name__=="__main__":
    allrows={}
    for site,C in [('wall',3.0),('wall',8.0),('dom',4.0),('dom',10.0),('AA',15.0),('AA',30.0)]:
        try: rows,sw,B,Z,hd=analyse(site,C)
        except FileNotFoundError: continue
        R=np.sqrt(C/theta)
        n_m=int((hd<0).sum()); err=np.median([r['err'] for r in rows]) if rows else np.nan
        print(f'{site:4s} C={C:4.1f}  Einstein R={R:5.1f}  swallow radius sqrt2*R={np.sqrt(2)*R:5.1f}: originals inside = {len(sw)} at beta={np.round(sorted(sw),1)}')
        print(f'       detected normal={int((hd>0).sum())} mirrored={n_m}; median |z - lens prediction| = {err:.2f}a')
        near=[r for r in rows if r['par']>0 and r['beta']<45]
        for r in sorted(near,key=lambda r:r['beta'])[:4]: print(f"         normal node beta={r['beta']:5.1f}: r={r['r']:5.1f} (pred {r['rpred']:5.1f}), rotation {r['dphi']:6.1f} deg (pred {r['dphipred']:6.1f})")
        for r in sorted([r for r in rows if r['par']<0],key=lambda r:r['r'])[:4]: print(f"         mirrored  beta={r['beta']:5.1f}: r={r['r']:5.1f} (pred {r['rpred']:5.1f}), rotation {r['dphi']:6.1f} deg (pred {r['dphipred']:6.1f})")
        allrows[f'{site}_{C}']=rows
    json.dump(allrows,open('lens_rows.json','w'))
