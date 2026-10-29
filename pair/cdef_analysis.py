"""Topological charges of the 8 single complex-charge defects (area removed/added, torque +/-; twist & mismatch).
Input: c_*.npz (from cdef.py), background none_* files. Output: cdef_topo.json (used by figCdef.py)."""
import json,numpy as np,os
from topo import analyse
th=1/30.; Ac=np.sqrt(3)/2; base=2*np.pi*th*4/Ac
cases=[('c_mm_def',0,4),('c_tw_def',90,4),('c_mm_add',0,-4),('c_tw_add',90,-4),('c_mm_tqp',0,4j),('c_mm_tqm',0,-4j),('c_tw_tqp',90,4j),('c_tw_tqm',90,-4j)]
bg={0:'none_d0.0_V0.005_a0_x_lam-0.5',90:'none_d0.0_V0.005_lam-0.5'}
res=[]
for f,a,C in cases:
    if not os.path.exists(f+'.npz'): continue
    o,_=analyse(f,[(8.,0.)]); p=o['cores'][0]
    phim=(np.degrees(np.angle(C))-a+180)%360-180
    E0=json.load(open(bg[a]+'.json'))['E']; E=json.load(open(f+'.json'))['E']
    res.append(dict(f=f,alpha=a,C=str(C),phim=phim,pred=base*(1-np.cos(np.radians(phim))),q=p['Qhole']-p['Nminus_local'],
                    absorbed=o['Qbox']-o['Nplus'],Q=p['Qhole'],Nm=p['Nminus_local'],z7=p['z7'],gauss=o['gauss'],dE=E-E0))
    print(res[-1])
json.dump(res,open('cdef_topo.json','w'))
