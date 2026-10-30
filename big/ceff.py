"""Effective (dressed) complex charge on circles: C_eff(rho) = -(1/2pi)[ oint du.n + i oint du.t ],  du = u - u_b(alpha)."""
import sys,os,numpy as np
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','pair'))
from conserv import circ, ub
def ceff(f,alpha,rs,center=(0.,0.),ref=None):
    D=np.load(f+'.npz'); x=D['x']; u0=ub(x,alpha)
    dx=D['ux']-u0[0]; dy=D['uy']-u0[1]
    if ref is not None:  # subtract background relaxation (periodic moire modulation)
        B=np.load(ref+'.npz'); dx=dx-(B['ux']-u0[0]); dy=dy-(B['uy']-u0[1])
    out=[]
    for r in rs:
        n=4000; t=np.linspace(0,2*np.pi,n,endpoint=False)
        a,b=circ(dx,dy,x,center[0],center[1],r,n)
        un=a*np.cos(t)+b*np.sin(t); ut=-a*np.sin(t)+b*np.cos(t)
        ds=2*np.pi*r/n
        out.append(-(un.sum()*ds+1j*ut.sum()*ds)/(2*np.pi))
    return np.array(out)
if __name__=='__main__':
    rs=np.arange(20,140.01,5.)
    for f,a,ref in [('B_tw_pull40',90,'B_bg_tw'),('B_tw_push40',90,'B_bg_tw')]:
        c=ceff(f,a,rs,ref=ref)
        print(f); print('  rho  ',' '.join(f'{r:6.0f}' for r in rs[::2])); print('  ReC  ',' '.join(f'{v.real:+6.2f}' for v in c[::2])); print('  ImC  ',' '.join(f'{v.imag:+6.2f}' for v in c[::2]))
        th=1/30.; Ac=np.sqrt(3)/2; k=th*np.exp(1j*np.radians(a))
        print('  -2pi/Ac Re(conj(k)C)',' '.join(f'{-2*np.pi/Ac*(np.conj(k)*v).real:+6.2f}' for v in c[::2]))
