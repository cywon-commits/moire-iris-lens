import sys,numpy as np
sys.path.insert(0,'/home/claude/cywon-commits/moire-iris-lens/pair'); sys.path.insert(0,'/home/claude/cywon-commits/moire-iris-lens/hex')
from pair4 import run
from conserv import Qint
from pscan_run import place,box,rh_of,kap
s3=np.sqrt(3); c1=np.array([0.5,1/(2*s3)]); th=1/30.; Ac=s3/2
alpha=float(sys.argv[1]); cabs=float(sys.argv[2]); phim=float(sys.argv[3]); pl=sys.argv[4]
Lp=box(cabs); rh=rh_of(cabs,'s'); z0=place(pl,alpha); C=cabs*np.exp(1j*np.radians(alpha+phim)); R0=np.sqrt(cabs/th); T=2*np.pi*th*cabs/Ac
Rr=run([(z0.real,z0.imag,C)],V0=0.0,alpha=alpha,lam=-0.5,Lp=Lp,rh=rh)
x=Rr['x']; X,Y=np.meshgrid(x,x,indexing='ij'); ub=c1[0]+1j*c1[1]+kap(alpha)*(X+1j*Y)
rs=np.arange(np.ceil(1.5*R0+10),Lp-15+1e-9,1.0)
dQ=np.array([Qint(Rr['ux'],Rr['uy'],x,z0.real,z0.imag,r)-Qint(ub.real,ub.imag,x,z0.real,z0.imag,r) for r in rs])
print(f"alpha={alpha:.0f} |C|={cabs:g} phim={phim:+.0f} {pl}: T={T:.2f} Tcos={T*np.cos(np.radians(phim)):+.2f}  elastic -dQ={-dQ.mean():+.2f}  ratio={-dQ.mean()/(T*np.cos(np.radians(phim))) if abs(np.cos(np.radians(phim)))>0.1 else float('nan'):.2f}",flush=True)
