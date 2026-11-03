"""S1-S4 analysis (tasks/TASK_screening_scan.md sec. 3). usage: python screen_ana.py JOBFILE [OUT.json]
Per defect run (job lines 'ALPHA CABS PHIM PLACE V0 LAM INIT LP'; 'bg' lines are skipped):
 1 rho_node : node-label tracking (nodedisp.node_positions on run and background), fit
              dz = m_eff/conj(w') + b0 + b1 w' + b2 conj(w'),  w' = relaxed node - z0 (exact for the linear lens: the node
              condition is evaluated at the displaced node; screen/mfit.py used the background node w = zb - z0, which biases
              rho low by O(|m|/w^2)), selected by 1.5R+5 < |zb - z0| < Lp-8; also in three equal radial shells. rho_node = m_eff/m (complex), m = C/kappa.
 2 rho_int, rho_area : -dQ and -dQ_area (conserv.Qint/Qarea) averaged over [1.5R+10, Lp-15], divided by T cos(phi_m).
 3 inner over-rotation / over-dilation: azimuthal means of d_omega/theta and d_div/theta (run minus background, core
   excluded) in rings r in [r_h+1, 2R] (width 2); b_bar = area (= r-weighted) mean over r_h+1 < r < 0.6R of d_omega
   (twist) or d_div/2 (mismatch), in units of theta.
 4 locked ring: fraction of each ring with V < 0.1 V0, run and background (V0 > 0 only).
 5 torque tau(r) = closed-circle integral of r t_t dl and radial traction P(r) = integral of r t_n dl
   (t = sigma n, sigma = lambda tr(eps) + 2 mu eps), background subtracted, r <= 3R; in units of tau0 = 4 pi mu |C|.
 6 E - E_bg, iterations."""
import sys,os,json,numpy as np
from scipy.ndimage import map_coordinates
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','hex')); from hexcore import Vhex
from nodedisp import node_positions
from conserv import Qint,Qarea
from pscan_run import place,rh_of,kap
from screen_run import btag,dtag
s3=np.sqrt(3.); Ac=s3/2; th=1/30.; mu=1.
def nodefit(D,B,alpha,z0,C,r1,r2,rmax):
    z,e=node_positions(D,alpha,rmax=rmax); zb,eb=node_positions(B,alpha,rmax=rmax)
    ok=(e<1e-6)&(eb<1e-6); d=z-zb; w=zb-z0; rr=np.abs(w); m=C/kap(alpha)
    wn=z-z0   # the node condition kappa z + du(z) = kappa zb gives d = -du(z)/kappa, i.e. the lens term is m_eff/conj(z - z0)
    def fit(a,b,ext=False):
        s=ok&(rr>a)&(rr<b)&(np.abs(d)<10)
        if s.sum()<(10 if ext else 6): return None
        w_=wn[s]; cols=[1/np.conj(w_),np.ones_like(w_),w_,np.conj(w_)]
        if ext: cols+=[1/w_,np.log(np.abs(w_)),w_/np.conj(w_)]   # shear force dipole (1/z) and net force on the core (Kelvin: log r, z/zbar)
        A=np.stack(cols,1); sol=np.linalg.lstsq(A,d[s],rcond=None)[0]
        rms=float(np.sqrt(np.mean(abs(d[s]-A@sol)**2))); r=sol[0]/m
        o=dict(n=int(s.sum()),rho=[r.real,r.imag],rho_abs=float(abs(r)),b0=[sol[1].real,sol[1].imag],b1=[sol[2].real,sol[2].imag],b2=[sol[3].real,sol[3].imag],rms=rms,r1=float(a),r2=float(b))
        if ext: o.update(dipole_1_over_z=[sol[4].real,sol[4].imag],force_log=[sol[5].real,sol[5].imag],force_z_over_zbar=[sol[6].real,sol[6].imag])
        return o
    edges=np.linspace(r1,r2,4)
    return fit(r1,r2),[fit(edges[i],edges[i+1]) for i in range(3)],fit(r1,r2,True),[fit(edges[i],edges[i+1],True) for i in range(3)]
def grads(E,h):
    ux,uy=E['ux'],E['uy']
    return np.gradient(ux,h,axis=0),np.gradient(ux,h,axis=1),np.gradient(uy,h,axis=0),np.gradient(uy,h,axis=1)
def circle_integrals(E,h,x,z0,rs,lam,n=4000):
    a,b,c,d=grads(E,h); exx,eyy,exy=a,d,0.5*(b+c); tr=exx+eyy; sxx=lam*tr+2*mu*exx; syy=lam*tr+2*mu*eyy; sxy=2*mu*exy
    t=np.linspace(0,2*np.pi,n,endpoint=False); nx,ny=np.cos(t),np.sin(t); tau=[]; P=[]
    for r in rs:
        I=(z0.real+r*nx-x[0])/h; J=(z0.imag+r*ny-x[0])/h; g=lambda A: map_coordinates(A,[I,J],order=1)
        trx=g(sxx)*nx+g(sxy)*ny; tr_y=g(sxy)*nx+g(syy)*ny; dl=2*np.pi*r/n
        tau.append(np.sum(r*(-trx*ny+tr_y*nx))*dl); P.append(np.sum(r*(trx*nx+tr_y*ny))*dl)
    return np.array(tau),np.array(P)
def one(job):
    alpha,cabs,phim=float(job[0]),float(job[1]),float(job[2]); pl=job[3]; V0,lam=float(job[4]),float(job[5]); init=job[6]; Lp=float(job[7])
    t=dtag(alpha,cabs,phim,pl,V0,lam,init,Lp); bt=btag(alpha,V0,lam,Lp)
    rec=dict(tag=t,alpha=alpha,absC=cabs,phim=phim,place=pl,V0=V0,lam=lam,K_over_mu=lam+1,init=init,Lp=Lp)
    if not os.path.exists(t+'.npz'): rec['status']='running' if os.path.exists(t+'_ck.npz') else 'missing'; return rec
    if not os.path.exists(bt+'.npz'): rec['status']='no background'; return rec
    D=np.load(t+'.npz'); B=np.load(bt+'.npz'); x=D['x']; h=x[1]-x[0]
    z0=place(pl,alpha); C=cabs*np.exp(1j*np.radians(alpha+phim)); R=np.sqrt(cabs/th); T=2*np.pi*th*cabs/Ac; Tc=T*np.cos(np.radians(phim)); rh=rh_of(cabs,'s')
    rec.update(status='converged',T=T,R=R,rh=rh,iters=int(D['tot']),dE=float(D['E'])-float(B['E']))
    full,shells,fx,fxs=nodefit(D,B,alpha,z0,C,1.5*R+5,Lp-8,Lp-8+15); rec['node']=full; rec['node_shells']=shells; rec['node_ext']=fx; rec['node_ext_shells']=fxs
    rec['rho_node']=full['rho'][0] if full else None; rec['rho_node_ext']=fx['rho'][0] if fx else None
    rs=np.arange(np.ceil(1.5*R+10),Lp-15+1e-9,1.0)
    dQ=np.array([Qint(D['ux'],D['uy'],x,z0.real,z0.imag,r)-Qint(B['ux'],B['uy'],x,z0.real,z0.imag,r) for r in rs])
    dA=np.array([Qarea(D['ux'],D['uy'],x,z0.real,z0.imag,r)-Qarea(B['ux'],B['uy'],x,z0.real,z0.imag,r) for r in rs])
    rec.update(mdQ=float(-dQ.mean()),mdA=float(-dA.mean()))
    if abs(Tc)>0.1*T: rec.update(rho_int=float(-dQ.mean()/Tc),rho_area=float(-dA.mean()/Tc))
    # profiles
    a,b,c,d=grads(D,h); ab,bb,cb,db=grads(B,h)
    dw=(0.5*(c-b)-0.5*(cb-bb))/th; ddiv=((a+d)-(ab+db))/th
    X,Y=np.meshgrid(x,x,indexing='ij'); rr=np.hypot(X-z0.real,Y-z0.imag); core=D['core']
    if V0>0: V,_,_=Vhex(D['ux'],D['uy'],V0,s3*V0); Vb,_,_=Vhex(B['ux'],B['uy'],V0,s3*V0)
    prof=[]
    for r0 in np.arange(rh+1,2*R,2.0):
        msk=(rr>=r0)&(rr<r0+2)&~core
        row=dict(r=float(r0+1),domega=float(dw[msk].mean()),ddiv=float(ddiv[msk].mean()))
        if V0>0: row.update(lock=float((V[msk]<0.1*V0).mean()),lock_bg=float((Vb[msk]<0.1*V0).mean()))
        prof.append(row)
    rec['profile']=prof
    msk=(rr>=rh+1)&(rr<0.6*R)&~core; bw=float(dw[msk].mean()); bd=float(ddiv[msk].mean())/2
    rec['b_bar']=bw if alpha==90 else bd; rec['b_omega']=bw; rec['b_div_half']=bd
    rec['theta_over_theta_plus_b']=1/(1+rec['b_bar'])
    rsT=np.arange(rh+1,min(3*R,Lp-10),2.0); tau,P=circle_integrals(D,h,x,z0,rsT,lam); taub,Pb=circle_integrals(B,h,x,z0,rsT,lam); tau0=4*np.pi*mu*cabs
    rec['torque']=dict(r=rsT.tolist(),tau=((tau-taub)/tau0).tolist(),P=((P-Pb)/tau0).tolist(),tau0=tau0)
    return rec
if __name__=='__main__':
    jobs=[l.split('#')[0].split() for l in open(sys.argv[1])]; jobs=[j for j in jobs if j and j[0]!='bg']
    out=sys.argv[2] if len(sys.argv)>2 else 'screen_scan.json'; res=[]
    for j in jobs:
        try: r=one(j)
        except Exception as e: r=dict(job=j,status='error',err=repr(e))
        res.append(r)
        if r.get('status')=='converged':
            nd=r['node']; print(f"{r['tag']:44s} rho_node={nd['rho'][0] if nd else float('nan'):+.3f}{nd['rho'][1] if nd else 0:+.3f}i ext={r['rho_node_ext'] if r['rho_node_ext'] is not None else float('nan'):+.3f} (n={nd['n'] if nd else 0}) rho_int={r.get('rho_int',float('nan')):.3f} rho_area={r.get('rho_area',float('nan')):.3f} b/θ={r['b_bar']:+.3f} θ/(θ+b)={r['theta_over_theta_plus_b']:.3f} dE={r['dE']:.2f}",flush=True)
    json.dump(res,open(out,'w'),indent=1)
