import sys,os,numpy as np
P='/home/claude/cywon-commits/moire-iris-lens/pair'; sys.path.insert(0,P); sys.path.insert(0,P+'/../hex')
os.chdir('/tmp/claude-0/half')
import pscan_run as pr
from pscan_run import place,kap
V0=float(sys.argv[1]); lam=float(os.environ.get("LAM","-0.5")); alpha=float(sys.argv[2]); cabs=float(sys.argv[3]); phim=float(sys.argv[4]); pl=sys.argv[5]; Lp=float(sys.argv[6])
bg=f"bg_l{lam:g}_a{alpha:.0f}_V{V0}_L{Lp:.0f}"
pr.relax(bg,dict(defects=[],V0=V0,alpha=alpha,lam=lam,Lp=Lp))
if cabs>0:
    z0=place(pl,alpha); C=cabs*np.exp(1j*np.radians(alpha+phim)); rh=2.4*np.sqrt(cabs/4)
    pr.relax(f'd_l{lam:g}_a{alpha:.0f}_V{V0}_C{cabs:g}_p{phim:+.0f}_{pl}_L{Lp:.0f}',dict(defects=[(z0.real,z0.imag,C)],V0=V0,alpha=alpha,lam=lam,Lp=Lp,rh=rh))
