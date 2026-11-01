"""Partition scan analysis (P1). usage: python pscan_ana.py JOBFILE [OUT.json]
For every job line 'ALPHA CABS PHIM PLACE INIT [RHMODE]' measures (topo.analyse conventions):
  q      = Q_hole - N_-  (N_- counted outside the core block within r_loc = 1.5R+10; q14 uses topo's default radius 14)
  q_lin  = same count on the synthetic linear lens field u_b - C/conj(z-z0) (unrelaxed background, same core mask)
  mdQ    = -Delta Q(rho) = -(Q_def - Q_bg) on circles about z0, averaged over rho in [1.5R+10, Lp-15]
  gauss  = Gauss-law residual from topo.analyse (must be 0), dE = E - E_bg, iterations
Unfinished runs are recorded with status 'running' (from the checkpoint) or 'missing'."""
import sys,os,json,tempfile,numpy as np
from topo import analyse,frac,cell_windings
from conserv import Qint
from pscan_run import place,box,rh_of,tag,bgtag,kap
s3=np.sqrt(3.); Ac=s3/2; th=1/30.; c1=np.array([0.5,1/(2*s3)])
TMP=os.environ.get('PSCAN_TMP') or tempfile.mkdtemp()
EPS=1e-4*(0.31+0.17j)   # tiny shift of the synthetic fields: keeps AA nodes off grid vertices (else windings are ambiguous)
def linear_field(x,alpha,z0,C,rh):
    X,Y=np.meshgrid(x,x,indexing='ij'); Z=X+1j*Y+EPS; k=kap(alpha); zz=Z-z0; R=np.abs(zz)
    u=c1[0]+1j*c1[1]+k*Z-C*zz/np.maximum(R,rh)**2
    return u.real,u.imag
def count(f,z0,rh,Lp,rloc):
    rq=rh+0.6; o,W=analyse(f,[(z0.real,z0.imag)],rq=rq,box=Lp-5.)
    D=np.load(f+'.npz'); x=D['x']; h=x[1]-x[0]; xm=0.5*(x[1:]+x[:-1]); XC,YC=np.meshgrid(xm,xm,indexing='ij')
    k=int(np.ceil(rq/h)); ci,cj=np.argmin(abs(x-z0.real)),np.argmin(abs(x-z0.imag))
    keep=np.ones_like(W,bool); keep[ci-k:ci+k,cj-k:cj+k]=False
    reg=keep&(np.hypot(XC-z0.real,YC-z0.imag)<rloc)
    p=o['cores'][0]
    return dict(Qhole=p['Qhole'],Nm=int((W[reg]<0).sum()),Np_loc=int((W[reg]>0).sum()),q=p['Qhole']-int((W[reg]<0).sum()),
                q14=p['Qhole']-p['Nminus_local'],z7=p['z7'],gauss=o['gauss'])
def one(job):
    alpha=float(job[0]); cabs=float(job[1]); phim=float(job[2]); pl=job[3]; init=job[4]; mode=job[5] if len(job)>5 else 's'
    Lp=box(cabs); rh=rh_of(cabs,mode); z0=place(pl,alpha); C=cabs*np.exp(1j*np.radians(alpha+phim))
    R=np.sqrt(cabs/th); T=2*np.pi*th*cabs/Ac; rloc=1.5*R+10; t=tag(alpha,cabs,phim,pl,init,mode)
    rec=dict(tag=t,alpha=alpha,absC=cabs,phim=phim,place=pl,z0=[z0.real,z0.imag],rh=rh,Lp=Lp,init=init,rhmode=mode,T=T,R=R,
             pred_q=T*(1-np.cos(np.radians(phim))),pred_far=T*np.cos(np.radians(phim)))
    f=t if os.path.exists(t+'.npz') else (t+'_ck' if os.path.exists(t+'_ck.npz') else None)
    if f is None: rec['status']='missing'; return rec
    rec['status']='converged' if f==t else 'running'
    D=np.load(f+'.npz'); B=np.load(bgtag(alpha,Lp)+'.npz'); x=D['x']
    rec.update(count(f,z0,rh,Lp,rloc)); rec['iters']=int(D['tot']); rec['dE']=float(D['E'])-float(B['E'])
    rs=np.arange(np.ceil(rloc),Lp-15+1e-9,1.0)
    dQ=np.array([Qint(D['ux'],D['uy'],x,z0.real,z0.imag,r)-Qint(B['ux'],B['uy'],x,z0.real,z0.imag,r) for r in rs])
    rec['rho']=rs.tolist(); rec['mdQ_curve']=(-dQ).tolist(); rec['mdQ']=float(-dQ.mean())
    lx,ly=linear_field(x,alpha,z0,C,rh); lf=os.path.join(TMP,'lin_'+t)
    np.savez(lf+'.npz',x=x,ux=lx,uy=ly); L=count(lf,z0,rh,Lp,rloc)
    k=kap(alpha); X,Y=np.meshgrid(x,x,indexing='ij'); Z=X+1j*Y+EPS; ub=c1[0]+1j*c1[1]+k*Z
    dQl=np.array([Qint(lx,ly,x,z0.real,z0.imag,r)-Qint(ub.real,ub.imag,x,z0.real,z0.imag,r) for r in rs])
    rec.update(q_lin=L['q'],Qhole_lin=L['Qhole'],gauss_lin=L['gauss'],mdQ_lin=float(-dQl.mean())); os.remove(lf+'.npz')
    return rec
if __name__=='__main__':
    jobs=[l.split('#')[0].split() for l in open(sys.argv[1])]; jobs=[j for j in jobs if j]
    out=sys.argv[2] if len(sys.argv)>2 else 'pscan_results.json'
    res=[]
    for j in jobs:
        try: r=one(j)
        except Exception as e: r=dict(job=j,status='error',err=repr(e))
        res.append(r)
        if r.get('status') in ('converged','running'):
            print(f"{r['tag']:36s} {r['status'][:4]} T={r['T']:.2f} q={r['q']} (q14 {r['q14']}, lin {r['q_lin']}) pred {r['pred_q']:.2f}  -dQ={r['mdQ']:+.2f} (lin {r['mdQ_lin']:+.2f}) pred {r['pred_far']:+.2f} gauss={r['gauss']} it={r['iters']}",flush=True)
    json.dump(res,open(out,'w'),indent=1)
