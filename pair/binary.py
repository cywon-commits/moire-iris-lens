import numpy as np
from skimage import measure
def detJ(Z,zs,ms):
    S=sum(m/np.conj(Z-zk)**2 for zk,m in zip(zs,ms)); return 1-np.abs(S)**2
def crit_curves(d,R=1.0,phase=-90.,L=4.,n=801):
    x=np.linspace(-L,L,n); X,Y=np.meshgrid(x,x,indexing='ij'); Z=X+1j*Y
    m=R**2*np.exp(1j*np.radians(phase)); zs=[-d/2,d/2]
    D=detJ(Z,zs,[m,m]); cs=measure.find_contours(D,0.0)
    cs=[c for c in cs if len(c)>20]
    return [ (x[0]+c[:,0]*(x[1]-x[0]))+1j*(x[0]+c[:,1]*(x[1]-x[0])) for c in cs ]
if __name__=="__main__":
    for d in [0.3,0.6,0.9,0.98,1.02,1.2,1.6,2.0,2.6,2.8,2.86,3.0,3.5]:
        print(f'd/R={d:4.2f}: number of critical curves = {len(crit_curves(d))}')
