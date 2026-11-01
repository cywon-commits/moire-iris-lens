import sys,os,json,numpy as np
from topo import frac, cell_windings
from conserv import Qint
s3=np.sqrt(3); Ac=s3/2; th=1/30.; C=120.; R=np.sqrt(C/th); nM=th**2/Ac; T=2*np.pi*th*C/Ac
def ana(f,bg,alpha,rhole=22.):
    D=np.load(f); B=np.load(bg); x=D['x']; s1,s2=frac(D['ux'],D['uy']); W=cell_windings(s1,s2)
    xm=0.5*(x[1:]+x[:-1]); XC,YC=np.meshgrid(xm,xm,indexing='ij'); Rr=np.hypot(XC,YC)
    Qh=Qint(D['ux'],D['uy'],x,0,0,rhole); kap=th*np.exp(1j*np.radians(alpha)); far=-(2*np.pi/Ac)*(np.conj(kap)*C).real
    out=dict(Qhole=Qh,T=T,rows=[])
    for rho in (40,55,70,85,100,115,130):
        m=(Rr>rhole)&(Rr<rho); Np=int((W[m]>0).sum()); Nm=int((W[m]<0).sum())
        Q=Qint(D['ux'],D['uy'],x,0,0,rho); Qb=Qint(B['ux'],B['uy'],B['x'],0,0,rho)
        out['rows'].append(dict(rho=rho,Nplus=Np,Nminus=Nm,Q=Q,Qbg=Qb,dNplus=Qb-Np,dQ=Q-Qb,q=Qh-Nm,
            pred_dNplus=T-np.pi*nM*R**4/rho**2 if rho>R else None,pred_dQ=far+np.pi*nM*R**4/rho**2 if rho>R else None))
    return out
if __name__=='__main__':
    f,bg,a=sys.argv[1],sys.argv[2],float(sys.argv[3]); o=ana(f,bg,a)
    print('Qhole',o['Qhole'],'T',round(T,2),'pred q =',round(T*(1-np.cos(np.radians(-90 if a==90 else 0))),2))
    for r in o['rows']: print({k:(round(v,2) if isinstance(v,float) else v) for k,v in r.items()})
    json.dump(o,open(os.path.basename(f).replace('.npz','')+'_ana.json','w'))
