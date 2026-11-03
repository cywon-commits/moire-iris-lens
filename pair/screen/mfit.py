import sys,numpy as np
sys.path.insert(0,'/home/claude/cywon-commits/moire-iris-lens/pair')
from nodedisp import node_positions
th=1/30.
def mfit(f,bg,alpha,z0,C,r1,r2,rmax=None):
    kap=th*np.exp(1j*np.radians(alpha))
    D=np.load(f+'.npz'); B=np.load(bg+'.npz'); L=float(D['x'].max())
    rmax=rmax or L-8
    z,e=node_positions(D,alpha,rmax=rmax+15); zb,eb=node_positions(B,alpha,rmax=rmax+15)
    ok=(e<1e-6)&(eb<1e-6); d=z-zb; w=zb-z0; rr=np.abs(w)
    sel=ok&(rr>r1)&(rr<r2)&(np.abs(d)<10)
    # model d = m/conj(w) + b0 + b1 w + b2 conj(w)
    w_=w[sel]; A=np.stack([1/np.conj(w_),np.ones_like(w_),w_,np.conj(w_)],1)
    sol,res,rk,sv=np.linalg.lstsq(A,d[sel],rcond=None)
    pred=A@sol; rms=np.sqrt(np.mean(abs(d[sel]-pred)**2))
    m=C/kap
    return sol[0], m, sel.sum(), rms, sol
if __name__=='__main__':
    pass
