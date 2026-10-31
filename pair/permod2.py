# moire elastic constants, oblique periodic cell spanned by strained moire vectors.
# registry u = c1 + P z + Q zbar + phi ; P = kap(1-Ua), Q = -kap Ub ; U = Ua z + Ub zbar (Eulerian moire displacement)
import sys,json,time,numpy as np
import os; sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','hex')); from hexcore import Vhex
from scipy.optimize import minimize
s3=np.sqrt(3.); c1=np.array([0.5,1/(2*s3)])
def solve(alpha,Ua,Ub,th=1/30.,V0=0.005,lam=-0.5,mu=1.,n=2,Ns=None,x0=None,seed=0,gtol=1e-11):
    kap=th*np.exp(1j*np.radians(alpha)); Ua=complex(Ua); Ub=complex(Ub)
    # unstrained moire vectors: kap*T0 = lattice vectors (1,0),(1/2,s3/2)
    T0=[n*1/kap, n*np.exp(1j*np.pi/3)/kap]
    # strained: (1-Ua)T - Ub conj(T) = T0  -> real 2x2 solve
    def strained(t0):
        a=1-Ua; b=-Ub   # a T + b Tbar = t0 ; T=x+iy
        M=np.array([[a.real+b.real, -a.imag+b.imag],[a.imag+b.imag, a.real-b.real]])
        v=np.linalg.solve(M,[t0.real,t0.imag]); return v[0]+1j*v[1]
    T=[strained(t) for t in T0]
    P=kap*(1-Ua); Q=-kap*Ub
    Ns=Ns or int(round(abs(T0[0])/0.45)); N1=N2=Ns
    e1=T[0]/N1; e2=T[1]/N2; Acell=abs((np.conj(T[0])*T[1]).imag); A=Acell/(N1*N2)
    i1,i2=np.meshgrid(np.arange(N1),np.arange(N2),indexing='ij'); Z=i1*e1+i2*e2
    W=P*Z+Q*np.conj(Z); bx=c1[0]+W.real; by=c1[1]+W.imag
    gW=(P+Q,1j*(P-Q)); gb=np.array([[gW[0].real,gW[1].real],[gW[0].imag,gW[1].imag]])  # gb[comp][dir]
    R=np.roll
    # triangle 0: edges e1,e2 from (i,j); triangle 1: edges -e1,-e2 from (i+1,j+1)
    def minv(a,b):
        M=np.array([[a.real,a.imag],[b.real,b.imag]]); return np.linalg.inv(M)   # g = Minv @ [da,db]
    Mi=[minv(e1,e2),minv(-e1,-e2)]
    def diffs(f):
        d0=(R(f,-1,0)-f, R(f,-1,1)-f)                       # from (i,j)
        f11=R(R(f,-1,0),-1,1); d1=(R(f,-1,1)-f11, R(f,-1,0)-f11)   # from (i+1,j+1): to (i,j+1)= -e1, to (i+1,j)= -e2
        return d0,d1
    def EG(z,parts=False):
        F=z.reshape(2,N1,N2); D=[diffs(F[0]),diffs(F[1])]; Eel=0.; Gr=np.zeros_like(F)
        for s in (0,1):
            g=[[Mi[s][0,0]*D[c][s][0]+Mi[s][0,1]*D[c][s][1]+gb[c][0], Mi[s][1,0]*D[c][s][0]+Mi[s][1,1]*D[c][s][1]+gb[c][1]] for c in (0,1)]
            exx=g[0][0]; eyy=g[1][1]; exy=0.5*(g[0][1]+g[1][0]); tr=exx+eyy
            Eel+=0.5*A*np.sum(mu*(exx**2+eyy**2+2*exy**2)+0.5*lam*tr**2)
            w=0.5*A; dg=[[w*(2*mu*exx+lam*tr), w*mu*2*exy],[w*mu*2*exy, w*(2*mu*eyy+lam*tr)]]
            for c in (0,1):
                da=Mi[s][0,0]*dg[c][0]+Mi[s][1,0]*dg[c][1]; db=Mi[s][0,1]*dg[c][0]+Mi[s][1,1]*dg[c][1]
                if s==0: Gr[c]+=R(da,1,0)-da+R(db,1,1)-db
                else:
                    # d1a = f(i,j+1)-f(i+1,j+1); d1b = f(i+1,j)-f(i+1,j+1)
                    Gr[c]+=R(da,1,1)+R(db,1,0)-R(R(da+db,1,0),1,1)
        V,gx,gy=Vhex(bx+F[0],by+F[1],V0,s3*V0); EV=A*np.sum(V); Gr[0]+=A*gx; Gr[1]+=A*gy
        Gr-=Gr.mean(axis=(1,2),keepdims=True)
        if parts: return Eel,EV
        return Eel+EV,Gr.ravel()
    if x0 is None: x0=1e-3*np.random.default_rng(seed).standard_normal(2*N1*N2)
    t=time.time()
    r=minimize(EG,x0,jac=True,method='L-BFGS-B',options=dict(maxiter=60000,maxfun=120000,gtol=gtol,ftol=1e-16,maxcor=30))
    Eel,EV=EG(r.x,True)
    return dict(alpha=alpha,Ua=[Ua.real,Ua.imag],Ub=[Ub.real,Ub.imag],e=float(r.fun/Acell),eel=float(Eel/Acell),eV=float(EV/Acell),nit=int(r.nit),t=time.time()-t,x=r.x,EG=EG)
if __name__=="__main__":
    # gradient check
    from scipy.optimize import check_grad
    R=solve(90,0.01+0.02j,0.015-0.01j,Ns=20,x0=None,gtol=1e-3)
    EG=R['EG']; z=np.random.default_rng(1).standard_normal(R['x'].size)*0.05
    print('gradcheck',check_grad(lambda q:EG(q)[0],lambda q:EG(q)[1],z), np.linalg.norm(EG(z)[1]))
