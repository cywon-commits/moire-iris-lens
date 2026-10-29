import os
import numpy as np, glob, sys
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','hex'))
from scipy.ndimage import map_coordinates
from matplotlib.path import Path
from affine import domain_field
s3=np.sqrt(3.); c1=np.array([0.5,1/(2*s3)])
def loopdata(f,margin=1.0,rad=None):
    D=np.load(f); x=D['x']; h=x[1]-x[0]; X0,Y0,k=float(D['X0']),float(D['Y0']),float(D['k'])
    hole=float(f.split('_h')[1][:-4]); r=(hole+margin) if rad is None else rad
    th=np.linspace(0,2*np.pi,4001)[:-1]; px=X0+r*np.cos(th); py=Y0+k*r*np.sin(th)
    idx=[(px-x[0])/h,(py-x[0])/h]
    ux=map_coordinates(D['ux'],idx,order=1); uy=map_coordinates(D['uy'],idx,order=1)
    f_=domain_field(D['ux'],D['uy']); v=map_coordinates(f_,idx,order=1); s=np.sign(v)
    walls=int((s!=np.roll(s,-1)).sum())
    P=Path(np.c_[ux,uy]); a1=np.array([1.,0]); a2=np.array([.5,s3/2])
    Q=sum(P.contains_point(c1+n1*a1+n2*a2) for n1 in range(-8,9) for n2 in range(-8,9))
    return hole,walls,Q
if __name__=="__main__":
    for case in ['cell31','cell30']:
        for f in sorted(glob.glob(f'nb_{case}_h*.npz'),key=lambda s: float(s.split('_h')[1][:-4])):
            hole,w,Q=loopdata(f); _,w0,Q0=loopdata(f,rad=max(float(f.split('_h')[1][:-4])+1,1.0))
            print(f'{case} pore {hole:4.1f}a : walls ending on loop just outside pore = {w:2d}, AA nodes inside (Q) = {Q}')
