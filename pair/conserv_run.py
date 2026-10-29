import numpy as np, json
from conserv import Qint, Qarea, ub
L='_lam-0.5'
bg={0:'none_d0.0_V0.005_a0_x'+L,90:'none_d0.0_V0.005'+L}
cases=[ # name, file, alpha, center, elastic file, complex factor on elastic field, family label
 ('mm def (C=4)','c_mm_def',0,(8.,0.),'el_s8',1,'single'),('mm add (C=-4)','c_mm_add',0,(8.,0.),'el_s8',-1,'single'),
 ('mm torque +','c_mm_tqp',0,(8.,0.),'el_s8',1j,'single'),('mm torque -','c_mm_tqm',0,(8.,0.),'el_s8',-1j,'single'),
 ('tw def','c_tw_def',90,(8.,0.),'el_s8',1,'single'),('tw add','c_tw_add',90,(8.,0.),'el_s8',-1,'single'),
 ('tw torque +','c_tw_tqp',90,(8.,0.),'el_s8',1j,'single'),('tw torque -','c_tw_tqm',90,(8.,0.),'el_s8',-1j,'single'),
 ('mm ++ d26 default','pair_d26.0_V0.005_a0_x'+L,0,(0.,0.),'el_pp26',1,'pp26'),('mm ++ d26 pull','mmout_d26.0',0,(0.,0.),'el_pp26',1,'pp26'),('mm ++ d26 push','mmin_d26.0',0,(0.,0.),'el_pp26',1,'pp26'),
 ('mm +- d26','opp_pm_d26.0_a0_V0.005',0,(0.,0.),'el_pm26',1,'pm26'),('tw +- d26','opp_pm_d26.0_a90_V0.005',90,(0.,0.),'el_pm26',1,'pm26'),
 ('tw ++ d40 default','pair_d40.0_V0.005'+L,90,(0.,0.),'el_pp40',1,'pp40'),('tw ++ d40 pull','swout_d40.0',90,(0.,0.),'el_pp40',1,'pp40'),('tw ++ d40 push','swin_d40.0',90,(0.,0.),'el_pp40',1,'pp40'),
 ('mm ++ d40','pair_d40.0_V0.005_a0_x'+L,0,(0.,0.),'el_pp40',1,'pp40'),
]
rs=np.arange(12,60.01,0.25); res={}
for name,f,a,c,el,fac,fam in cases:
    D=np.load(f+'.npz'); B=np.load(bg[a]+'.npz'); E=np.load(el+'.npz'); x=D['x']
    u0=ub(x,a); e0=ub(x,0); du=(E['ux']-e0[0])+1j*(E['uy']-e0[1]); du=fac*du
    lx,ly=u0[0]+du.real,u0[1]+du.imag
    row=dict(r=rs.tolist(),dQ=[],dQarea=[],dQlin=[],fam=fam,file=f)
    for r in rs:
        qb=Qint(B['ux'],B['uy'],x,*c,r); row['dQ'].append(Qint(D['ux'],D['uy'],x,*c,r)-qb)
        row['dQarea'].append(Qarea(D['ux'],D['uy'],x,*c,r)-Qarea(B['ux'],B['uy'],x,*c,r))
        row['dQlin'].append(Qarea(lx,ly,x,*c,r)-Qarea(*u0,x,*c,r))
    res[name]=row
    dq=np.array(row['dQ']); dl=np.array(row['dQlin'])
    win=[(24,36),(36,48),(48,58)]
    print(f'{name:20s} '+'  '.join(f'[{a_}-{b_}] meas {dq[(rs>=a_)&(rs<b_)].mean():+.2f} lin {dl[(rs>=a_)&(rs<b_)].mean():+.2f}' for a_,b_ in win),flush=True)
json.dump(res,open('conserv_res.json','w'))
