import os
import numpy as np, sys, json
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','hex'))
from iris_an import nodes, node_handedness
from gl_an import images, sources, theta
refs={0:'gl_a0_C0.0.npz',45:'gl_a45_C0.0.npz',90:'lens_wall_C0.0.npz',180:'gl_a180_C0.0.npz'}
def rel(f,Rmax=52):
    D=np.load(f); x=D['x']; C=float(D['C']); al=float(D['alpha']); o=D['u0']; m=C*np.exp(-1j*np.radians(al))/theta; R=np.sqrt(abs(m))
    Dr=np.load(refs[int(al)]); Pr=nodes(x,Dr['ux'],Dr['uy'],Dr['core']); Zr=Pr[:,0]+1j*Pr[:,1]
    P=nodes(x,D['ux'],D['uy'],D['core']); hd=node_handedness(x,D['ux'],D['uy'],P); Z=P[:,0]+1j*P[:,1]
    Zn=Z[(hd>0)&(abs(Z)<Rmax)]; Zm=Z[(hd<0)&(abs(Z)<Rmax)]
    out=[]
    for zr in Zr[abs(Zr)<Rmax]:
        ims=images(zr,m)                     # lens applied to the relaxed reference position
        nor=[z for z,p in ims if p>0]
        if not nor:
            hit=[z for z in Zn if abs(z-zr)<0.35*abs(zr)]
            out.append(dict(beta=abs(zr),swallowed_pred=True,found=len(hit)>0)); continue
        zp=nor[0]; k=np.argmin(abs(Zn-zp)) if len(Zn) else None
        if k is None or abs(Zn[k]-zp)>0.5*abs(zr): out.append(dict(beta=abs(zr),swallowed_pred=False,found=False)); continue
        z=Zn[k]
        out.append(dict(beta=abs(zr),swallowed_pred=False,found=True,r=abs(z),rpred=abs(zp),dphi=float(np.degrees(np.angle(z/zr))),dphipred=float(np.degrees(np.angle(zp/zr)))))
    return dict(C=C,alpha=al,phim=float(np.degrees(np.angle(m))),R=float(R),nodes=out,mirror_r=sorted([round(float(abs(z)),1) for z in Zm]))
if __name__=="__main__":
    res={}
    for f in ['gl_a0_C4.0.npz','gl_a45_C4.0.npz','gl_a90_C4.0.npz','gl_a90_C-4.0.npz','gl_a180_C4.0.npz']:
        A=rel(f); res[f]=A
        print(f"{f}: phi_m={A['phim']:6.1f}")
        for n in sorted(A['nodes'],key=lambda n:n['beta']):
            if n['swallowed_pred']: print(f"   ref node at {n['beta']:5.1f}: predicted SWALLOWED -> {'still present' if n['found'] else 'gone (ok)'}")
            elif n['found']: print(f"   ref node at {n['beta']:5.1f}: r {n['r']:5.1f} (pred {n['rpred']:5.1f})   rotation {n['dphi']:6.1f} (pred {n['dphipred']:6.1f})")
            else: print(f"   ref node at {n['beta']:5.1f}: not matched")
        print('   mirrored nodes at r =',A['mirror_r'])
    json.dump(res,open('gl_rel.json','w'))
