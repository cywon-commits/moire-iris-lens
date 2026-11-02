"""P2 (tasks/TASK_partition_scan.md section 3): big single lens with a chosen initial state and defect centre.
usage: python big_ck2.py ALPHA CRE CIM LP RH INIT PLACE TAG
  INIT lin = linear lens field (as big_ck.py), bg = relaxed defect-free background pbg_a{ALPHA}_L{LP}.npz
  PLACE AA (centre z=0, as big_ck.py) or AB/SP/GEN (pscan_run.place)
Checkpointed like big_ck.py (400-iteration chunks, stop when |dE| < 1e-5)."""
import sys,os,numpy as np
from pscan_run import relax,place,bgtag
a=float(sys.argv[1]); C=complex(float(sys.argv[2]),float(sys.argv[3])); Lp=float(sys.argv[4]); rh=float(sys.argv[5])
init=sys.argv[6]; pl=sys.argv[7]; tag=sys.argv[8]
z0=place(pl,a); x0=None
if init=='bg':
    B=np.load(bgtag(a,Lp)+'.npz'); x0=np.concatenate([B['ux'].ravel(),B['uy'].ravel()])
relax(tag,dict(defects=[(z0.real,z0.imag,C)],V0=0.005,alpha=a,lam=-0.5,Lp=Lp,rh=rh),x0=x0)
