import os
import numpy as np, sys
from scipy.ndimage import gaussian_filter, map_coordinates
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','hex'))
from affine import domain_field
from hexcore import Vhex
s3=np.sqrt(3)
def wrinkle(f):
    D=np.load(f); x=D['x']; h=D['h']; ux,uy=D['ux'],D['uy']; xc,yc=float(D['xc']),float(D['yc'])
    d1=float(D['d1']); dd=d1*Vhex(ux,uy,1.,s3)[0]/9.0
    r=h-dd                       # height relative to the local stacking-dependent equilibrium
    X,Y=np.meshgrid(x-xc,x-yc,indexing='ij'); R=np.hypot(X,Y)
    out=[]
    for r0,r1 in [(6,9),(9,13),(13,20),(25,40)]:
        m=(R>r0)&(R<r1)&(h<0.8); out.append(round(float(np.sqrt(np.mean(r[m]**2))),4))
    return out
if __name__=="__main__":
    for f in sys.argv[1:]: print(f,'rms of h - d(u) in rings (6-9,9-13,13-20,25-40 a):',wrinkle(f))
