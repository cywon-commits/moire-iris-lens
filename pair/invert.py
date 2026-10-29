import os
import numpy as np, sys
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','hex')); sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','nb'))
from scipy.optimize import least_squares
from scipy.ndimage import map_coordinates
from iris_an import nodes, node_handedness
s3=np.sqrt(3.); c1=np.array([0.5,1/(2*s3)]); A1=1.0+0j; A2=0.5+1j*s3/2
def observe(f,Rmax=60):
    """'measurement': node positions, handedness, and their AA index (what unwrapping by cell counting gives)"""
    D=np.load(f); x=D['x']; P=nodes(x,D['ux'],D['uy'],D['core']); P=P[np.hypot(P[:,0],P[:,1])<Rmax]
    hd=node_handedness(x,D['ux'],D['uy'],P); h=x[1]-x[0]
    ux=map_coordinates(D['ux'],[(P[:,0]-x[0])/h,(P[:,1]-x[0])/h],order=1); uy=map_coordinates(D['uy'],[(P[:,0]-x[0])/h,(P[:,1]-x[0])/h],order=1)
    w=(ux-c1[0])+1j*(uy-c1[1])               # u - c1 = n1*A1 + n2*A2 (+ small error)
    n2=np.round(w.imag/(s3/2)); n1=np.round(w.real-0.5*n2); AA=c1[0]+1j*c1[1]+n1*A1+n2*A2
    return P[:,0]+1j*P[:,1],hd,AA
def model(p,z,K):
    o=p[0]+1j*p[1]; kap=p[2]+1j*p[3]; lam=p[4]+1j*p[5]; u=o+kap*z+lam*np.conj(z)
    for k in range(K):
        zk=p[6+3*k]+1j*p[7+3*k]; C=p[8+3*k]; u=u-C/np.conj(z-zk)
    return u
def fit(z,AA,K,kap0,starts=None,rng=None,ntry=40):
    rng=rng or np.random.default_rng(0); best=None
    for t in range(ntry):
        p=[(AA-kap0*z).mean().real,(AA-kap0*z).mean().imag,kap0.real,kap0.imag,0.,0.]
        for k in range(K):
            if starts is not None and t==0: zk=starts[k]
            else: zk=rng.uniform(-40,40)+1j*rng.uniform(-40,40)
            p+=[zk.real,zk.imag,rng.uniform(1,8)]
        lo=[-np.inf]*6+[-70,-70,0.0]*K; hi=[np.inf]*6+[70,70,30.]*K
        try:
            res=least_squares(lambda q: np.concatenate([(model(q,z,K)-AA).real,(model(q,z,K)-AA).imag]),p,bounds=(lo,hi),method='trf',max_nfev=3000)
        except ValueError: continue
        if best is None or res.cost<best.cost: best=res
    return best
def summarize(best,K,n):
    p=best.x; out=[(round(p[6+3*k],2),round(p[7+3*k],2),round(p[8+3*k],2)) for k in range(K)]
    rms=np.sqrt(2*best.cost/n); bic=2*n*np.log(rms**2+1e-30)+ (6+3*K)*np.log(2*n)
    return out,rms,bic,p
if __name__=="__main__":
    kap0=1j/30
    for f,truth in [('R_d8.0_V0.005.npz',[(4,0,4)]),('R_d26.0_V0.005.npz',[(13,0,4)]),('R_d50.0_V0.005.npz',[(25,0,4)]),
                    ('pair_d12.0_V0.005.npz',[(-6,0,4),(6,0,4)]),('pair_d32.0_V0.005.npz',[(-16,0,4),(16,0,4)]),('pair_d50.0_V0.005.npz',[(-25,0,4),(25,0,4)])]:
        z,hd,AA=observe(f); print(f'{f}: {len(z)} nodes ({(hd>0).sum()} normal, {(hd<0).sum()} mirrored); truth {truth}')
        prev=None
        for K in (0,1,2,3,4):
            b=fit(z,AA,K,kap0); out,rms,bic,p=summarize(b,K,len(z))
            keep=prev is None or (bic<prev[1]-10 and min(c for _,_,c in out)>0.5)
            print(f'   K={K}: rms {rms:.4f}a  BIC {bic:8.1f}  {"accept" if keep else "reject"}  deficits (x, y, C) = {out}')
            if not keep: break
            prev=(K,bic)
