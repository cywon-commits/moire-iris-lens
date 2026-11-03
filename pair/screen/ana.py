import sys,numpy as np
P='/home/claude/cywon-commits/moire-iris-lens/pair'; sys.path.insert(0,P); sys.path.insert(0,P+'/../hex')
from conserv import Qint,Qarea
from pscan_run import place
sys.path.insert(0,'/tmp/claude-0/half'); from mfit import mfit
th=1/30.; Ac=np.sqrt(3)/2
def ana(dfile,bgfile,alpha,cabs,phim,pl):
    D=np.load(dfile+'.npz'); B=np.load(bgfile+'.npz'); x=D['x']; Lp=x.max()
    z0=place(pl,alpha); R=np.sqrt(cabs/th); T=2*np.pi*th*cabs/Ac; Tc=T*np.cos(np.radians(phim))
    rs=np.arange(np.ceil(1.5*R+10),Lp-15+1e-9,1.0)
    dQ=np.array([Qint(D['ux'],D['uy'],x,z0.real,z0.imag,r)-Qint(B['ux'],B['uy'],x,z0.real,z0.imag,r) for r in rs])
    dA=np.array([Qarea(D['ux'],D['uy'],x,z0.real,z0.imag,r)-Qarea(B['ux'],B['uy'],x,z0.real,z0.imag,r) for r in rs])
    C=cabs*np.exp(1j*np.radians(alpha+phim))
    me,m,n,rms,_=mfit(dfile,bgfile,alpha,z0,C,1.5*R+5,Lp-6)
    return dict(T=T,Tc=Tc,mdQ=-dQ.mean(),mdA=-dA.mean(),ratio_int=-dQ.mean()/Tc,ratio_area=-dA.mean()/Tc,mratio=(me/m),nn=n,rms=rms)
if __name__=='__main__':
    for V0 in ('0.0025','0.005','0.01','0.02'):
        o=ana(f'd_a90_V{V0}_C8_p+0_GEN_L72',f'bg_a90_V{V0}_L72',90,8,0,'GEN')
        print(V0,{k:(np.round(v,3) if not isinstance(v,(int,np.integer)) else v) for k,v in o.items()})
