"""P4 (tasks/TASK_partition_scan.md): domain-wall width from relaxed defect-free backgrounds (box ±72).
usage: python wallwidth.py [V0 ...]   (default 0.0025 0.005 0.01 0.02; twist and mismatch)
Cut: through the wall midpoint z_m = (a1/2)/kappa (saddle SP between AA nodes 0 and a1/kappa), perpendicular to the wall
(wall tangent = direction of a1/kappa).  Across this wall u - c_AA crosses the line c_AA + t a1, so the jump is the
partial Burgers vector b = a/sqrt3 perpendicular to a1 in u-space: twist -> b parallel to the wall (shear wall),
mismatch -> b perpendicular to the wall (tensile wall).
u_par(s) = (u - c_AA).e_b along the cut, fitted on |s| < 13 (half the spacing of parallel walls) with
u_par = A + B tanh((s - s0)/w)  ->  w_tanh.  Also the full width at half maximum of V(s) - min V on the same window.
Output: wallwidth.json."""
import sys,os,json,numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import curve_fit
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','hex'))
from hexcore import Vhex
from pscan_run import relax,kap
s3=np.sqrt(3.); c1=np.array([0.5,1/(2*s3)]); a1=np.array([1.,0.])
def bg(alpha,V0,Lp=72.):
    t=f'pbg_a{alpha:.0f}_L{Lp:.0f}'+('' if V0==0.005 else f'_V{V0:g}')
    relax(t,dict(defects=[],V0=V0,alpha=alpha,lam=-0.5,Lp=Lp)); return t
def cut(t,alpha,V0,half=13.,n=521):
    D=np.load(t+'.npz'); x=D['x']; h=x[1]-x[0]; k=kap(alpha)
    zm=0.5/k; tw=(1/k)/abs(1/k); nrm=1j*tw; s=np.linspace(-half,half,n); z=zm+s*nrm
    I=(z.real-x[0])/h; J=(z.imag-x[0])/h
    ux=map_coordinates(D['ux'],[I,J],order=3); uy=map_coordinates(D['uy'],[I,J],order=3)
    eb=np.array([0.,1.])                                    # unit vector perpendicular to a1 in u-space
    upar=(ux-c1[0])*eb[0]+(uy-c1[1])*eb[1]
    f=lambda s,A,B,s0,w: A+B*np.tanh((s-s0)/w)
    p,_=curve_fit(f,s,upar,p0=[upar.mean(),(upar[-1]-upar[0])/2,0.,3.]); A,B,s0,w=p
    V,_,_=Vhex(ux,uy,V0,s3*V0); Vr=V-V.min(); hm=Vr.max()/2; above=s[Vr>=hm]
    # wall type: angle between Burgers direction (u-space) and physical wall tangent mapped by the background
    btan=abs(np.dot(eb,[tw.real,tw.imag]))
    return dict(alpha=alpha,V0=V0,w_tanh=abs(w),B=B,s0=s0,step=2*abs(B),fwhm_V=float(above.max()-above.min()),Vmax=float(Vr.max()),
                rms_fit=float(np.sqrt(np.mean((f(s,*p)-upar)**2))),wall='shear' if btan>0.5 else 'tensile',iters=int(D['tot']))
if __name__=='__main__':
    V0s=[float(v) for v in sys.argv[1:]] or [0.0025,0.005,0.01,0.02]
    res=[]
    for a in (90.,0.):
        for V0 in V0s:
            r=cut(bg(a,V0),a,V0); res.append(r); print(r,flush=True)
    mu,lam=1.,-0.5
    fit={}
    for a,K in ((90.,mu),(0.,lam+2*mu)):
        rr=[r for r in res if r['alpha']==a]; X=np.sqrt(K/np.array([r['V0'] for r in rr]))
        for key in ('w_tanh','fwhm_V'):
            Y=np.array([r[key] for r in rr]); c=float(np.dot(X,Y)/np.dot(X,X)); p=np.polyfit(np.log([r['V0'] for r in rr]),np.log(Y),1)
            fit[f'a{a:.0f}_{key}']=dict(K=K,coef=c,law=f'{key} = {c:.3f} sqrt(K/V0)',loglog_slope=float(p[0]))
    print(json.dumps(fit,indent=1))
    json.dump(dict(runs=res,fit=fit,note='K=mu for the shear (twist) wall, K=lambda+2mu for the tensile (mismatch) wall; lengths in a=1, L_M=30'),open('wallwidth.json','w'),indent=1)
