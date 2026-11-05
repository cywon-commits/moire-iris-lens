"""Local-density (LDA) theory of lens-mass screening.
Axisymmetric extra rotation (twist) or extra half-dilation (mismatch) d(r) on top of the lens field.
A(r) = |C| - 2 int_{rh}^r d(s) s ds  (exact Stokes/flux identity), far coefficient C_eff = A(inf).
Local G: P = theta + d (rotation for twist, dilation for mismatch), Q = |d + A/r^2| (shear).
Energy density e(G) = e_hom + 3 V0 g(Q/|P|, V0 theta^2/P^2) from S0 (pair/eG.json)."""
import json,numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import minimize
D=json.load(open('/home/claude/cywon-commits/moire-iris-lens/pair/eG.json'))['runs']
th=1/30.; mu=1.0
RS=np.array([0,0.25,0.5,0.9,1.1,1.5,2,3,5.]); VS=np.array([0.0025,0.005,0.01,0.02])
def make_g(alpha,lam):
    G=np.zeros((len(RS),len(VS)))
    for i,r in enumerate(RS):
        for j,v in enumerate(VS):
            xs=[x for x in D if x['alpha']==alpha and abs(x['lam']-lam)<1e-9 and abs(x['V0']-v)<1e-12 and abs(x['r']-r)<1e-9]
            G[i,j]=np.mean([(x['e']-x['e_hom'])/(3*v) for x in xs])
    # extend in r with tail g = 1-(1-g5)*5/r, in logV linearly (then clip)
    rf=np.r_[RS,np.geomspace(6,400,40)]; Gf=np.vstack([G,1-(1-G[-1])[None,:]*5/np.geomspace(6,400,40)[:,None]])
    lv=np.log(VS); lvf=np.linspace(np.log(1e-5),np.log(1.0),60)
    Gv=np.array([[np.interp(l,lv,row) if lv[0]<=l<=lv[-1] else (row[0]+(l-lv[0])*(row[1]-row[0])/(lv[1]-lv[0]) if l<lv[0] else row[-1]+(l-lv[-1])*(row[-1]-row[-2])/(lv[-1]-lv[-2])) for l in lvf] for row in Gf])
    Gv=np.clip(Gv,0,1)
    I=RegularGridInterpolator((rf,lvf),Gv,bounds_error=False,fill_value=None)
    return lambda r,v: np.clip(I(np.stack([np.clip(r,0,400),np.clip(np.log(v),lvf[0],lvf[-1])],-1)),0,1)
class LDA:
    def __init__(s,bg,lam,V0,C=8.,rh=None,L=150.,n=48):
        s.bg=bg; s.lam=lam; s.K=lam+mu; s.V0=V0; s.C=C; s.rh=rh or 2.4*np.sqrt(C/4); s.L=L
        s.g=make_g(90. if bg=='twist' else 0.,lam)
        s.edges=np.geomspace(s.rh,L,n+1); s.n=n
        s.r=np.geomspace(s.rh,L,4000); s.rm=0.5*(s.r[1:]+s.r[:-1]); s.dA=np.pi*(s.r[1:]**2-s.r[:-1]**2)
        s.idx=np.clip(np.searchsorted(s.edges,s.rm)-1,0,n-1)
    def fields(s,x):
        d=x[s.idx]*th
        A=s.C-np.cumsum(2*d*s.rm*np.diff(s.r)); A=np.r_[s.C,A][:-1]*0.5+A*0.5
        P=th+d; Q=np.abs(d+A/s.rm**2)
        return d,A,P,Q
    def E(s,x):
        d,A,P,Q=s.fields(x)
        ehom=2*mu*Q**2+(2*s.K*(P**2-th**2) if s.bg=='mismatch' else 0.)
        Pa=np.maximum(np.abs(P),1e-6); ereg=3*s.V0*s.g(Q/Pa,s.V0*th**2/Pa**2)
        return np.sum((ehom+ereg)*s.dA)
    def solve(s,x0=None):
        x0=np.zeros(s.n) if x0 is None else x0
        r=minimize(s.E,x0,method='L-BFGS-B',options=dict(maxiter=3000,eps=1e-6))
        d,A,P,Q=s.fields(r.x)
        return dict(rho=A[-1]/s.C,x=r.x,E=r.fun,E0=s.E(np.zeros(s.n)),
                    bbar=np.sum(d[s.rm<0.6*np.sqrt(s.C/th)]*s.rm[s.rm<0.6*np.sqrt(s.C/th)])/np.sum(s.rm[s.rm<0.6*np.sqrt(s.C/th)])/th)

class LDA2(LDA):
    """LDA with the global constraint (clamped far boundary) imposed through the background
    'pre-stress' multiplier c0 = de/dP at (theta,0), and d(r) restricted to r < rmax."""
    def __init__(s,bg,lam,V0,C=8.,rh=None,rmax_over_R=3.,n=36,**kw):
        R=np.sqrt(C/th); LDA.__init__(s,bg,lam,V0,C,rh,L=rmax_over_R*R,n=n)
        h=1e-4*th
        s.c0=(s.e_loc(np.array([th+h]),np.array([0.]))-s.e_loc(np.array([th-h]),np.array([0.])))[0]/(2*h)
        s.e00=s.e_loc(np.array([th]),np.array([0.]))[0]
    def e_loc(s,P,Q):
        ehom=2*mu*Q**2+(2*s.K*(P**2) if s.bg=='mismatch' else 0.)
        Pa=np.maximum(np.abs(P),1e-6); return ehom+3*s.V0*s.g(Q/Pa,s.V0*th**2/Pa**2)
    def E(s,x):
        d,A,P,Q=s.fields(x)
        e=s.e_loc(P,Q)-s.e00-s.c0*(P-th)
        # outside rmax: pure lens tail with coefficient A_end, shear A_end/r^2 (elastic + registry change vs background)
        Aend=A[-1]; rt=np.geomspace(s.L,40*s.L,400); rtm=0.5*(rt[1:]+rt[:-1]); dAt=np.pi*np.diff(rt**2)
        Qt=Aend/rtm**2; et=s.e_loc(np.full_like(Qt,th),Qt)-s.e00
        return np.sum(e*s.dA)+np.sum(et*dAt)
