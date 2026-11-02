"""P3 (tasks/TASK_partition_scan.md sec. 4): twist lens with uniform heterostrain, u_b = c_AA + kappa z + eta conj(z), eta = r*theta*e^{i arg}.
usage: python hetero_run.py CABS PHIM R ARG PLACE      (defect run, linear Chang-Refsdal start, r_h = 2.4 sqrt(|C|/4))
       python hetero_run.py bg LP R ARG                 (defect-free background with the same heterostrain)
Box as P1 (|C|<=8: 72, <=16: 108, else 150). eta=0 runs are the P1 files and are not repeated."""
import sys,numpy as np
from pscan_run import relax,place,box,rh_of
th=1/30.; A=90.
def eta_of(r,arg): return r*th*np.exp(1j*np.radians(arg))
def htag(cabs,phim,r,arg,pl): return f'het_C{cabs:g}_p{phim:+.0f}_r{r:g}_e{arg:.0f}_{pl}'
def hbg(Lp,r,arg): return f'hbg_L{Lp:.0f}_r{r:g}_e{arg:.0f}'
if __name__=='__main__':
    if sys.argv[1]=='bg':
        Lp,r,arg=float(sys.argv[2]),float(sys.argv[3]),float(sys.argv[4])
        relax(hbg(Lp,r,arg),dict(defects=[],V0=0.005,alpha=A,lam=-0.5,Lp=Lp,eta=eta_of(r,arg))); sys.exit()
    cabs,phim,r,arg,pl=float(sys.argv[1]),float(sys.argv[2]),float(sys.argv[3]),float(sys.argv[4]),sys.argv[5]
    Lp=box(cabs); z0=place(pl,A); C=cabs*np.exp(1j*np.radians(A+phim))
    relax(htag(cabs,phim,r,arg,pl),dict(defects=[(z0.real,z0.imag,C)],V0=0.005,alpha=A,lam=-0.5,Lp=Lp,rh=rh_of(cabs,'s'),eta=eta_of(r,arg)))
