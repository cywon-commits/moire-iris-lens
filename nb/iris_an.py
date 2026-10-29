import os
import numpy as np, sys
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','hex'))
from scipy.ndimage import maximum_filter, gaussian_filter, map_coordinates
from affine import domain_field
from hexcore import Vhex
s3=np.sqrt(3.); theta=1/12
def nodes(x,ux,uy,core):
    V=Vhex(ux,uy,1.,s3)[0]; V[core]=0
    pk=(V==maximum_filter(V,size=7))&(V>7.0); pk[:6]=pk[-6:]=False; pk[:,:6]=pk[:,-6:]=False
    I,J=np.nonzero(pk); return np.array([(x[i],x[j]) for i,j in zip(I,J)])
def jac_sign(x,ux,uy,core,sig):
    h=x[1]-x[0]; a=gaussian_filter(ux,sig/h); b=gaussian_filter(uy,sig/h)
    uxx,uxy=np.gradient(a,h); uyx,uyy=np.gradient(b,h); J=uxx*uyy-uxy*uyx; J[core]=np.nan; return J
def node_handedness(x,ux,uy,P,r=1.2):
    """winding of the AB/BA sector sequence around each node: +1 same handedness as far-field moire, -1 mirrored"""
    h=x[1]-x[0]; out=[]
    for (px,py) in P:
        t=np.linspace(0,2*np.pi,721)[:-1]
        idx=[(px+r*np.cos(t)-x[0])/h,(py+r*np.sin(t)-x[0])/h]
        vx=map_coordinates(ux,idx,order=1); vy=map_coordinates(uy,idx,order=1)
        # winding of u around the AA point it encircles
        c=np.array([0.5,1/(2*s3)]); a1=np.array([1.,0]); a2=np.array([.5,s3/2])
        m=np.array([vx.mean(),vy.mean()]); best=None
        for n1 in range(-60,61):
            for n2 in range(-60,61):
                q=c+n1*a1+n2*a2
                d=np.hypot(*(q-m))
                if best is None or d<best[0]: best=(d,q)
        q=best[1]; ang=np.unwrap(np.arctan2(vy-q[1],vx-q[0])); out.append(int(np.round((ang[-1]-ang[0]+(ang[1]-ang[0]))/(2*np.pi))))
    return np.array(out)
if __name__=="__main__":
    for C in [4.0,10.0,20.0,35.0]:
        D=np.load(f'iris_C{C}.npz'); x=D['x']; core=D['core']
        for tag,(ux,uy) in [('rigid',(D['ux0'],D['uy0'])),('relaxed',(D['ux'],D['uy']))]:
            P=nodes(x,ux,uy,core); R=np.hypot(P[:,0],P[:,1]) if len(P) else np.array([])
            rs=np.sqrt(C/theta); hand=node_handedness(x,ux,uy,P[R<30]) if len(P) else []
            Rin=R[R<30]
            n_in=int(((Rin<rs))&True).sum() if False else int((Rin<rs).sum())
            neg=int((np.array(hand)<0).sum()); pos=int((np.array(hand)>0).sum())
            neg_in=int(((np.array(hand)<0)&(Rin<rs)).sum())
            print(f'C={C:4.0f} {tag:7s} r*={rs:4.1f}  nodes r<30: {len(Rin):3d}  inside r*: {n_in:3d}  mirrored(-1): {neg:3d} (inside r*: {neg_in})  normal(+1): {pos}')
