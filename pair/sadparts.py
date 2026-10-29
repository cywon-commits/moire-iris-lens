import os
import numpy as np, sys
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','hex')); from hexcore import Vhex
from scipy.ndimage import binary_dilation
s3=np.sqrt(3)
def parts(ux,uy,x,core,lam,mu=1.):
    h=x[1]-x[0]; exx=np.gradient(ux,h,axis=0); eyy=np.gradient(uy,h,axis=1); exy=0.5*(np.gradient(ux,h,axis=1)+np.gradient(uy,h,axis=0))
    tr=exx+eyy; dev=(exx-eyy)**2/2+2*exy**2; m=~binary_dilation(core,iterations=int(3/h)); V,_,_=Vhex(ux,uy,0.005,s3*0.005)
    return np.array([h*h*np.sum((0.5*(mu+lam)*tr**2)[m]),h*h*np.sum((mu*dev)[m]),h*h*np.sum(V[m])])
for tag,lam,src in [('L05_d40b',-0.5,'swout_d40.0'),('L2_d40b',1.0,'twL1out_d40.0')]:
    D=np.load(f'neb_{tag}.npz'); N=int(D['N']); x=D['x']; core=np.load(src+'.npz')['core']; R=D['Rall']; i=int(D['imax'])
    pA=parts(*R[0].reshape(2,N,N),x,core,lam); pS=parts(*R[i].reshape(2,N,N),x,core,lam)
    print(tag,'saddle-A: dil %.3f dev %.3f reg %.3f (sum %.3f) ; total barrier %.3f'%(*(pS-pA),(pS-pA).sum(),D['E'][i]-D['E'][0]))
