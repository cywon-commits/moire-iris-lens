"""(+,-) vs (+,+) iris pairs: elastic and moire-mediated interaction, node charges. Output: opp_results.json (figOpp.py)."""
import json,numpy as np
from topo import analyse
J=lambda t: json.load(open(t+'.json'))['E']; L='_lam-0.5'
bg0=J('none_d0.0_V0.0_a0_x'+L); bgV={0:J('none_d0.0_V0.005_a0_x'+L),90:J('none_d0.0_V0.005'+L)}
Rdef={0:lambda d:J(f'R_d{d}_V0.005_a0_x'+L),90:lambda d:J(f'R_d{d}_V0.005'+L)}
ppE={0:lambda d:J(f'pair_d{d}_V0.005_a0_x'+L),90:lambda d:J(f'pair_d{d}_V0.005'+L)}
out=[]
for d in [8.,16.,26.,40.]:
    U0pm=J(f'opp_pm_d{d}_a0_V0.0')-J(f'R_d{d}_V0.0_a0_x'+L)-J(f'opp_Radd_d{d}_a0_V0.0')+bg0
    U0pp=J(f'pair_d{d}_V0.0_a0_x'+L)-2*J(f'R_d{d}_V0.0_a0_x'+L)+bg0
    row=dict(d=d,U0pm=U0pm,U0pp=U0pp)
    for a in (0,90):
        row[f'Upm{a}']=J(f'opp_pm_d{d}_a{a}_V0.005')-Rdef[a](d)-J(f'opp_Radd_d{d}_a{a}_V0.005')+bgV[a]-U0pm
        row[f'Upp{a}']=ppE[a](d)-2*Rdef[a](d)+bgV[a]-U0pp
        o,_=analyse(f'opp_pm_d{d}_a{a}_V0.005',[(-d/2,0.),(d/2,0.)]); oa,_=analyse(f'opp_Radd_d{d}_a{a}_V0.005',[(d/2,0.)])
        row[f'abs_pm{a}']=o['Qbox']-o['Nplus']; row[f'q_pm{a}']=[p['Qhole']-p['Nminus_local'] for p in o['cores']]
        row[f'abs_add{a}']=oa['Qbox']-oa['Nplus']; row[f'gauss{a}']=o['gauss']
    out.append(row); print(row)
json.dump(out,open('opp_results.json','w'))
