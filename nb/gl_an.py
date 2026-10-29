import os
import numpy as np, sys, json
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','hex'))
from iris_an import nodes, node_handedness
theta=1/40; s3=np.sqrt(3); c1=np.array([0.5,1/(2*s3)]); a1=np.array([1.,0]); a2=np.array([.5,s3/2])
def images(beta,m):
    b=abs(beta)**2+2*m.real; disc=b*b-4*abs(m)**2
    if disc<0 or b<=0: return []
    out=[]
    for sg in (1,-1):
        r=np.sqrt((b+sg*np.sqrt(disc))/2); phi=np.angle(beta)-np.angle(r-m/r)
        out.append((r*np.exp(1j*phi),1 if r*r>abs(m) else -1))
    return out
def sources(o,alpha,rmax=12):
    B=[]
    for n1 in range(-14,15):
        for n2 in range(-14,15):
            q=c1+n1*a1+n2*a2-o; rho=q[0]+1j*q[1]
            if 1e-9<abs(rho)<rmax: B.append(rho/(theta*np.exp(1j*np.radians(alpha))))
    return np.array(B)
def analyse(f,Rmax=55):
    D=np.load(f); x=D['x']; C=float(D['C']); al=float(D['alpha']); o=D['u0']
    m=C*np.exp(-1j*np.radians(al))/theta; R=np.sqrt(abs(m)); phim=np.degrees(np.angle(m))
    B=sources(o,al); pred=[(b,)+im for b in B for im in images(b,m)]
    P=nodes(x,D['ux'],D['uy'],D['core']); P=P[np.hypot(P[:,0],P[:,1])<Rmax]; hd=node_handedness(x,D['ux'],D['uy'],P)
    rows=[]
    for (px,py),par in zip(P,hd):
        z=px+1j*py; cand=[p for p in pred if p[2]==par]
        if not cand: continue
        k=int(np.argmin([abs(z-p[1]) for p in cand])); b,zp,_=cand[k]
        rows.append(dict(beta=abs(b),r=abs(z),rpred=abs(zp),dphi=float(np.degrees(np.angle(z/b))),dphipred=float(np.degrees(np.angle(zp/b))),par=int(par),err=float(abs(z-zp))))
    bs=2*R*abs(np.sin(np.radians(phim)/2)); sw=sorted([round(abs(b),1) for b in B if abs(b)<bs])
    return dict(C=C,alpha=al,phim=float(phim),R=float(R),swallow_radius=float(bs),swallowed=sw,n_norm=int((hd>0).sum()),n_mirr=int((hd<0).sum()),rows=rows)
if __name__=="__main__":
    out={}
    for f in sys.argv[1:]:
        A=analyse(f); out[f]=A; rows=A['rows']
        nn=[r for r in rows if r['par']>0 and r['beta']<45]; mm=[r for r in rows if r['par']<0]
        print(f"{f}: phi_m={A['phim']:6.1f}  R={A['R']:.1f}  swallow radius={A['swallow_radius']:.1f} -> swallowed sources {A['swallowed']}")
        print(f"   detected normal {A['n_norm']}, mirrored {A['n_mirr']}; median position error {np.median([r['err'] for r in rows]):.2f}a")
        for r in sorted(nn,key=lambda r:r['beta'])[:3]: print(f"   normal beta={r['beta']:5.1f}: r={r['r']:5.1f} (pred {r['rpred']:5.1f}) rot {r['dphi']:6.1f} (pred {r['dphipred']:6.1f})")
    json.dump(out,open('gl_rows.json','w'))
