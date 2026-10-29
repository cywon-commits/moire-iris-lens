import os
import numpy as np,json,sys,glob
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','hex')); from hexcore import Vhex
s3=np.sqrt(3)
def tether(f):
    D=np.load(f+'.npz'); x=D['x']; d=json.load(open(f+'.json'))['d'] if 'd' in json.load(open(f+'.json')) else float(f.split('_d')[1])
    j0=np.argmin(abs(x)); sel=np.abs(x)<d/2-6
    V,_,_=Vhex(D['ux'][sel,j0-3:j0+4],D['uy'][sel,j0-3:j0+4],0.005,s3*0.005)
    return d,float(V.max(axis=1).min())   # weakest point along the axis of the max-over-band registry
if __name__=='__main__':
    for f in sys.argv[1:]:
        d,v=tether(f); print(f'{f:22s} d={d:4.0f}  min_x max_y V on axis = {v:.4f}  ({"tether" if v>0.01 else "no tether"})')
