# Linear drive of the eigen-inclusion, exact (envelope theorem): E1 = dE/deps* at eps*=0 = [F(u0;+d)-F(u0;-d)]/2d, u0 = baseline
import sys,numpy as np
from pair4 import run
def E1(base,alpha,Lp,r0,x0=0.,y0=0.,d=1e-4):
    D=np.load(base+'.npz'); z=np.concatenate([D['ux'].ravel(),D['uy'].ravel()])
    out=[]
    for s in (d,-d):
        EG,N,x=run([],V0=0.005,alpha=alpha,lam=-0.5,Lp=Lp,incl=(x0,y0,r0,s),want_EG=True); out.append(EG(z)[0])
    return (out[0]-out[1])/(2*d)
if __name__=='__main__':
    base=sys.argv[1]; a=float(sys.argv[2]); Lp=float(sys.argv[3])
    for r0 in [float(v) for v in sys.argv[4:]]: print(r0, E1(base,a,Lp,r0), flush=True)
