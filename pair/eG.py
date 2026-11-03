"""S0 (tasks/TASK_screening_scan.md): energy density e(G) of a periodic moire with uniform displacement gradient
u = c_AA + P z + Q conj(z) + phi(z),  P = kappa = theta e^{i alpha} (twist 90 / mismatch 0),  Q = eta = r theta e^{i arg}.
Same discretisation and energy as permod2.solve, but the periodic cell is the Gauss-reduced basis of the strained moire
lattice {T : P T + Q conj(T) in Lambda} (n x n moire cells, n=1 default: agrees with n=2 to 1e-5 relative at eta/theta=0 and 0.5) and each direction gets its own grid count (spacing ~h), so
strongly sheared (long, anisotropic) moires stay resolved. Singular values of G: sigma1,2 = |P| +- |Q|; det G = |P|^2-|Q|^2.
usage: python eG.py ALPHA R ARG V0 LAM [OUT.json]   (one point; appends to OUT, default eG_runs/<tag>.json)"""
import sys,os,json,time,numpy as np
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','hex')); from hexcore import Vhex
from scipy.optimize import minimize
s3=np.sqrt(3.); c1=np.array([0.5,1/(2*s3)])
def reduce_basis(a,b):
    while True:
        if abs(b)<abs(a): a,b=b,a
        m=round((b*np.conj(a)).real/abs(a)**2)
        if m==0: return a,b
        b=b-m*a
def cell(P,Q,n):
    def solveT(L):   # P T + Q conj(T) = L  (real 2x2)
        M=np.array([[P.real+Q.real,-P.imag+Q.imag],[P.imag+Q.imag,P.real-Q.real]]); v=np.linalg.solve(M,[L.real,L.imag]); return v[0]+1j*v[1]
    return reduce_basis(solveT(n*1.0),solveT(n*np.exp(1j*np.pi/3)))
def solve(alpha,r,arg,V0=0.005,lam=-0.5,mu=1.,th=1/30.,n=1,h=0.45,gtol=1e-10,seed=0,maxit=60000):
    P=th*np.exp(1j*np.radians(alpha)); Q=r*th*np.exp(1j*np.radians(arg))
    T1,T2=cell(P,Q,n); N1=max(12,int(np.ceil(abs(T1)/h))); N2=max(12,int(np.ceil(abs(T2)/h)))
    e1=T1/N1; e2=T2/N2; Acell=abs((np.conj(T1)*T2).imag); A=Acell/(N1*N2)
    i1,i2=np.meshgrid(np.arange(N1),np.arange(N2),indexing='ij'); Z=i1*e1+i2*e2
    W=P*Z+Q*np.conj(Z); bx=c1[0]+W.real; by=c1[1]+W.imag
    gW=(P+Q,1j*(P-Q)); gb=np.array([[gW[0].real,gW[1].real],[gW[0].imag,gW[1].imag]])
    R=np.roll
    def minv(a,b): return np.linalg.inv(np.array([[a.real,a.imag],[b.real,b.imag]]))
    Mi=[minv(e1,e2),minv(-e1,-e2)]
    def diffs(f):
        d0=(R(f,-1,0)-f,R(f,-1,1)-f); f11=R(R(f,-1,0),-1,1); d1=(R(f,-1,1)-f11,R(f,-1,0)-f11); return d0,d1
    def EG(z,parts=False):
        F=z.reshape(2,N1,N2); D=[diffs(F[0]),diffs(F[1])]; Eel=0.; Gr=np.zeros_like(F)
        for s in (0,1):
            g=[[Mi[s][0,0]*D[c][s][0]+Mi[s][0,1]*D[c][s][1]+gb[c][0],Mi[s][1,0]*D[c][s][0]+Mi[s][1,1]*D[c][s][1]+gb[c][1]] for c in (0,1)]
            exx=g[0][0]; eyy=g[1][1]; exy=0.5*(g[0][1]+g[1][0]); tr=exx+eyy
            Eel+=0.5*A*np.sum(mu*(exx**2+eyy**2+2*exy**2)+0.5*lam*tr**2)
            w=0.5*A; dg=[[w*(2*mu*exx+lam*tr),w*mu*2*exy],[w*mu*2*exy,w*(2*mu*eyy+lam*tr)]]
            for c in (0,1):
                da=Mi[s][0,0]*dg[c][0]+Mi[s][1,0]*dg[c][1]; db=Mi[s][0,1]*dg[c][0]+Mi[s][1,1]*dg[c][1]
                if s==0: Gr[c]+=R(da,1,0)-da+R(db,1,1)-db
                else: Gr[c]+=R(da,1,1)+R(db,1,0)-R(R(da+db,1,0),1,1)
        V,gx,gy=Vhex(bx+F[0],by+F[1],V0,s3*V0); EV=A*np.sum(V); Gr[0]+=A*gx; Gr[1]+=A*gy
        Gr-=Gr.mean(axis=(1,2),keepdims=True)
        if parts: return Eel,EV
        return Eel+EV,Gr.ravel()
    z0=np.zeros(2*N1*N2); E0el,E0V=EG(z0,True)
    x0=1e-3*np.random.default_rng(seed).standard_normal(2*N1*N2); t=time.time()
    res=minimize(EG,x0,jac=True,method='L-BFGS-B',options=dict(maxiter=maxit,maxfun=2*maxit,gtol=gtol,ftol=1e-16,maxcor=30))
    Eel,EV=EG(res.x,True)
    # homogeneous elastic energy density of G alone (rotation costs nothing)
    exx=gb[0][0]; eyy=gb[1][1]; exy=0.5*(gb[0][1]+gb[1][0]); tr=exx+eyy; ehom=mu*(exx**2+eyy**2+2*exy**2)+0.5*lam*tr**2   # same density as EG: mu eps:eps + lam/2 (tr eps)^2
    s1,s2=abs(P)+abs(Q),abs(abs(P)-abs(Q))
    return dict(alpha=alpha,r=r,arg=arg,V0=V0,lam=lam,K_over_mu=(lam+mu)/mu,theta=th,P=[P.real,P.imag],Q=[Q.real,Q.imag],
                detG=float(abs(P)**2-abs(Q)**2),sigma1=float(s1),sigma2=float(s2),
                e=float(res.fun/Acell),eel=float(Eel/Acell),eV=float(EV/Acell),e_unrelaxed=float((E0el+E0V)/Acell),e_hom=float(ehom),
                T1=[T1.real,T1.imag],T2=[T2.real,T2.imag],N=[N1,N2],Acell=float(Acell),nit=int(res.nit),gnorm=float(np.abs(res.jac).max()),
                success=bool(res.success),msg=str(res.message),t=round(time.time()-t,1))
def tag(alpha,r,arg,V0,lam): return f'eG_a{alpha:.0f}_r{r:g}_g{arg:.0f}_V{V0:g}_l{lam:g}'
if __name__=='__main__':
    a,r,g,V0,lam=(float(v) for v in sys.argv[1:6])
    os.makedirs('eG_runs',exist_ok=True); out=sys.argv[6] if len(sys.argv)>6 else os.path.join('eG_runs',tag(a,r,g,V0,lam)+'.json')
    if os.path.exists(out): print(out,'done already'); sys.exit()
    d=solve(a,r,g,V0,lam); json.dump(d,open(out,'w'),indent=1)
    print(tag(a,r,g,V0,lam),'e',d['e'],'e_unrel',d['e_unrelaxed'],'N',d['N'],'nit',d['nit'],'t',d['t'],flush=True)
