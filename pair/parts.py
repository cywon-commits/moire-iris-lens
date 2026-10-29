import os
import numpy as np, sys, json
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','hex')); from hexcore import Vhex
s3=np.sqrt(3)
def parts(f,lam,alpha,mu=1.,V0=0.005):
    D=np.load(f+'.npz'); x=D['x']; h=x[1]-x[0]; ux,uy=D['ux'],D['uy']; X,Y=np.meshgrid(x,x,indexing='ij')
    th=1/30.; k=th*np.exp(1j*np.radians(alpha))
    exx=np.gradient(ux,h,axis=0); eyy=np.gradient(uy,h,axis=1); exy=0.5*(np.gradient(ux,h,axis=1)+np.gradient(uy,h,axis=0))
    tr=exx+eyy; dev=(exx-eyy)**2/2+2*exy**2
    from scipy.ndimage import binary_dilation
    m=~binary_dilation(D['core'],iterations=int(3/h))
    V,_,_=Vhex(ux,uy,V0,s3*V0)
    return dict(dil=float(h*h*np.sum((0.5*(mu+lam)*tr**2)[m])),dev=float(h*h*np.sum((mu*dev)[m])),reg=float(h*h*np.sum(V[m])))
if __name__=='__main__':
    lam=float(sys.argv[1]); alpha=float(sys.argv[2])
    for f in sys.argv[3:]:
        p=parts(f,lam,alpha); print(f'{f:28s} dil={p["dil"]:8.3f} dev={p["dev"]:8.3f} reg={p["reg"]:8.3f} sum={sum(p.values()):8.3f}  E={json.load(open(f+".json"))["E"]:.3f}')
