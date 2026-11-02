"""P3 analysis (twist heterostrain lens). usage: python hetero_ana.py JOBFILE [OUT.json]
Same measurements as pscan_ana.py (q, q_lin on the linear Chang-Refsdal field u_b(eta) - C/conj(z-z0), -dQ window mean vs the
heterostrained background, Gauss residual, dE). eta=0 rows are taken from the P1 runs (lower-energy init)."""
import sys,os,json,tempfile,numpy as np
from conserv import Qint
from pscan_ana import count,EPS
from pscan_run import place,box,rh_of,tag,bgtag,kap
from hetero_run import eta_of,htag,hbg
s3=np.sqrt(3.); Ac=s3/2; th=1/30.; c1=np.array([0.5,1/(2*s3)]); A=90.
TMP=os.environ.get('PSCAN_TMP') or tempfile.mkdtemp()
def fields(x,z0,C,rh,eta):
    X,Y=np.meshgrid(x,x,indexing='ij'); Z=X+1j*Y+EPS; zz=Z-z0; R=np.abs(zz)
    ub=c1[0]+1j*c1[1]+kap(A)*Z+eta*np.conj(Z); return ub,ub-C*zz/np.maximum(R,rh)**2
def one(cabs,phim,r,arg,pl):
    Lp=box(cabs); rh=rh_of(cabs,'s'); z0=place(pl,A); C=cabs*np.exp(1j*np.radians(A+phim)); eta=eta_of(r,arg)
    R=np.sqrt(cabs/th); T=2*np.pi*th*cabs/Ac; rloc=1.5*R+10
    rec=dict(absC=cabs,phim=phim,r=r,arg=arg,place=pl,T=T,pred_q=T*(1-np.cos(np.radians(phim))),pred_far=T*np.cos(np.radians(phim)),Lp=Lp)
    if r==0:
        cand=[tag(A,cabs,phim,pl,i) for i in 'ab' if os.path.exists(tag(A,cabs,phim,pl,i)+'.npz')]
        if not cand: rec['status']='missing'; return rec
        f=min(cand,key=lambda t:float(np.load(t+'.npz')['E'])); fb=bgtag(A,Lp)
    else:
        f=htag(cabs,phim,r,arg,pl); fb=hbg(Lp,r,arg)
        if not os.path.exists(f+'.npz'): rec['status']='running' if os.path.exists(f+'_ck.npz') else 'missing'; return rec
        if not os.path.exists(fb+'.npz'): rec['status']='no background'; return rec
    rec.update(status='converged',file=f)
    D=np.load(f+'.npz'); B=np.load(fb+'.npz'); x=D['x']
    rec.update(count(f,z0,rh,Lp,rloc)); rec['iters']=int(D['tot']); rec['dE']=float(D['E'])-float(B['E'])
    rs=np.arange(np.ceil(rloc),Lp-15+1e-9,1.0)
    rec['mdQ']=float(-np.mean([Qint(D['ux'],D['uy'],x,z0.real,z0.imag,q)-Qint(B['ux'],B['uy'],x,z0.real,z0.imag,q) for q in rs]))
    ub,ul=fields(x,z0,C,rh,eta); lf=os.path.join(TMP,'hl_'+f); np.savez(lf+'.npz',x=x,ux=ul.real,uy=ul.imag); L=count(lf,z0,rh,Lp,rloc); os.remove(lf+'.npz')
    rec.update(q_lin=L['q'],gauss_lin=L['gauss'],
               mdQ_lin=float(-np.mean([Qint(ul.real,ul.imag,x,z0.real,z0.imag,q)-Qint(ub.real,ub.imag,x,z0.real,z0.imag,q) for q in rs])))
    return rec
if __name__=='__main__':
    res=[]
    for c in (4,24):
        for r,args in ((0,(0,)),(0.1,(0,60)),(0.3,(0,60))):
            for arg in args:
                for ph in (0,180):
                    for pl in ('AA','GEN'):
                        try: rec=one(c,ph,r,arg,pl)
                        except Exception as e: rec=dict(absC=c,phim=ph,r=r,arg=arg,place=pl,status='error',err=repr(e))
                        res.append(rec)
                        if rec.get('status')=='converged':
                            print(f"|C|={c} phi={ph:3d} r={r} arg={arg:2d} {pl:3s} q={rec['q']} lin={rec['q_lin']} pred={rec['pred_q']:.2f} -dQ={rec['mdQ']:+.2f} lin={rec['mdQ_lin']:+.2f} pred={rec['pred_far']:+.2f} gauss={rec['gauss']}",flush=True)
    json.dump(res,open(sys.argv[2] if len(sys.argv)>2 else 'hetero_results.json','w'),indent=1)
