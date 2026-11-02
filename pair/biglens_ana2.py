"""P2: biglens_ana.ana generalised to a defect centre z0 (pscan_run.place) and any background file; also the energy.
usage: python biglens_ana2.py RUN.npz BG.npz ALPHA PLACE [CRE CIM RH LP]  -> prints rows, writes <RUN>_ana.json"""
import sys,os,json,numpy as np
from topo import frac, cell_windings
from conserv import Qint
from pscan_run import place
s3=np.sqrt(3); Ac=s3/2; th=1/30.
def ana(f,bg,alpha,z0,C=120.,rhole=22.):
    D=np.load(f); B=np.load(bg); x=D['x']; W=cell_windings(*frac(D['ux'],D['uy'])); R=np.sqrt(abs(C)/th); nM=th**2/Ac; T=2*np.pi*th*abs(C)/Ac
    xm=0.5*(x[1:]+x[:-1]); XC,YC=np.meshgrid(xm,xm,indexing='ij'); Rr=np.hypot(XC-z0.real,YC-z0.imag)
    Qh=Qint(D['ux'],D['uy'],x,z0.real,z0.imag,rhole); kap=th*np.exp(1j*np.radians(alpha)); far=-(2*np.pi/Ac)*(np.conj(kap)*C).real
    out=dict(Qhole=Qh,T=T,z0=[z0.real,z0.imag],rows=[])
    for rho in (40,55,70,85,100,115,130):
        m=(Rr>rhole)&(Rr<rho); Np=int((W[m]>0).sum()); Nm=int((W[m]<0).sum())
        Q=Qint(D['ux'],D['uy'],x,z0.real,z0.imag,rho); Qb=Qint(B['ux'],B['uy'],B['x'],z0.real,z0.imag,rho)
        out['rows'].append(dict(rho=rho,Nplus=Np,Nminus=Nm,Q=Q,Qbg=Qb,dNplus=Qb-Np,dQ=Q-Qb,q=Qh-Nm,
            pred_dNplus=T-np.pi*nM*R**4/rho**2 if rho>R else None,pred_dQ=far+np.pi*nM*R**4/rho**2 if rho>R else None))
    # node-free radius: distance from z0 to the nearest positive node outside the hole
    P=Rr[(W>0)&(Rr>rhole)]; out['node_free_radius']=float(P.min()) if P.size else None
    return out
def energy(f,alpha,z0,C,rh,Lp):
    from pair4 import run
    EG,N,x=run([(z0.real,z0.imag,C)],V0=0.005,alpha=alpha,lam=-0.5,Lp=Lp,rh=rh,want_EG=True)
    D=np.load(f); return float(EG(np.concatenate([D['ux'].ravel(),D['uy'].ravel()]))[0])
if __name__=='__main__':
    f,bg,a,pl=sys.argv[1],sys.argv[2],float(sys.argv[3]),sys.argv[4]
    C=complex(float(sys.argv[5]),float(sys.argv[6])) if len(sys.argv)>6 else 120+0j; rh=float(sys.argv[7]) if len(sys.argv)>7 else 20.; Lp=float(sys.argv[8]) if len(sys.argv)>8 else 150.
    z0=place(pl,a); o=ana(f,bg,a,z0,C); o['E']=energy(f,a,z0,C,rh,Lp); Bz=np.load(bg); o['E_bg']=float(Bz['E']) if 'E' in Bz.files else None
    print('Qhole',o['Qhole'],'T',round(o['T'],2),'E',round(o['E'],3),'E_bg',o['E_bg'],'node-free radius',o['node_free_radius'])
    for r in o['rows']: print({k:(round(float(v),2) if isinstance(v,(float,np.floating)) else v) for k,v in r.items()})
    json.dump(o,open(os.path.basename(f).replace('.npz','')+'_ana.json','w'),indent=1,default=float)
