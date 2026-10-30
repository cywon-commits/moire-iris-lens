"""Task 1: far node charge vs integer branch in the large box. Output: big_res.json"""
import sys,os,json,numpy as np
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','pair'))
from conserv import Qint, Qarea, ub
from topo import analyse
cases=[('tw pull d40','B_tw_pull40',90,[(-20.,0.),(20.,0.)],'B_el_pp40','B_bg_tw'),
       ('tw push d40','B_tw_push40',90,[(-20.,0.),(20.,0.)],'B_el_pp40','B_bg_tw'),
       ('mm pull d26','B_mm_pull26',0,[(-13.,0.),(13.,0.)],'B_el_pp26','B_bg_mm'),
       ('mm push d26','B_mm_push26',0,[(-13.,0.),(13.,0.)],'B_el_pp26','B_bg_mm')]
rs=np.arange(20,140.01,0.5); res={}
for name,f,a,cores,el,bg in cases:
    if not all(os.path.exists(g+'.npz') for g in (f,el,bg)): print('skip',name); continue
    D=np.load(f+'.npz'); B=np.load(bg+'.npz'); E=np.load(el+'.npz'); x=D['x']
    u0=ub(x,a); e0=ub(x,0); lx=u0[0]+E['ux']-e0[0]; ly=u0[1]+E['uy']-e0[1]
    o,_=analyse(f,cores,box=145.); ob,_=analyse(bg,[],box=145.)
    row=dict(r=rs.tolist(),dQ=[],dQarea=[],dQlin=[],E=json.load(open(f+'.json'))['E'],
             q=[p['Qhole']-p['Nminus_local'] for p in o['cores']],absorbed=o['Qbox']-o['Nplus'],gauss=o['gauss'],Qbox=o['Qbox'],Qbox_bg=ob['Qbox'])
    for r in rs:
        row['dQ'].append(Qint(D['ux'],D['uy'],x,0.,0.,r)-Qint(B['ux'],B['uy'],x,0.,0.,r))
        row['dQarea'].append(Qarea(D['ux'],D['uy'],x,0.,0.,r)-Qarea(B['ux'],B['uy'],x,0.,0.,r))
        row['dQlin'].append(Qarea(lx,ly,x,0.,0.,r)-Qarea(*u0,x,0.,0.,r))
    res[name]=row
    dq=np.array(row['dQ']); dl=np.array(row['dQlin'])
    print(f"{name}: E={row['E']:.3f} q={row['q']} absorbed={row['absorbed']} gauss={row['gauss']} Qbox={row['Qbox']}/{row['Qbox_bg']} | "+
          '  '.join(f'[{a_}-{b_}] {dq[(rs>=a_)&(rs<b_)].mean():+.2f} (lin {dl[(rs>=a_)&(rs<b_)].mean():+.2f})' for a_,b_ in [(20,50),(50,80),(80,110),(110,140)]),flush=True)
json.dump(res,open('big_res.json','w'))
