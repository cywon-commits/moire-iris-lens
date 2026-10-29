import json,numpy as np,matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from collections import Counter
R=json.load(open('topo_all.json')); ZW=json.load(open('zw.json'))
col={-1:'#8e44ad',0:'#7f8c8d',1:'#e67e22',2:'#27ae60',3:'#2471a3'}
fig=plt.figure(figsize=(16,9.6))
for k,(fam,lam,title) in enumerate([('twist',-0.5,'(a) twist, K/μ=0.5: every converged pair state'),('mismatch',-0.5,'(b) mismatch, K/μ=0.5: every converged pair state')]):
    ax=fig.add_subplot(2,3,k+1); P=[r for r in R if r['fam']==fam and r['mode']=='pair' and r['lam']==lam]
    ds=sorted(set(r['d'] for r in P)); Emin={d:min(r['E'] for r in P if r['d']==d) for d in ds}
    seen=set()
    for r in P:
        a=r['Qbox']-r['Nplus']; lab=f'absorbed charge {a:+d}' if a not in seen else None; seen.add(a)
        ax.scatter(r['d'],r['E']-Emin[r['d']],s=55,color=col[a],edgecolor='k',lw=.4,label=lab,zorder=3)
    ax.set_xlabel('d'); ax.set_ylabel('E − min_branch E(d)'); ax.set_title(title,fontsize=10)
    h,l=ax.get_legend_handles_labels(); o=np.argsort(l); ax.legend([h[i] for i in o],[l[i] for i in o],fontsize=8)
    ax.text(0.02,0.97,'ideal (lens): 2×0.97 = 1.94' if fam=='twist' else 'ideal (lens): 0 (all far-field)',transform=ax.transAxes,va='top',fontsize=9)
ax=fig.add_subplot(2,3,3)
for fam,off,c in [('twist',-0.18,'#e67e22'),('mismatch',0.18,'#7f8c8d')]:
    q=[p['Qhole']-p['Nminus_local'] for r in R if r['fam']==fam and r['mode'] in ('R','pair') for p in r['cores']]
    C=Counter(q); xs=sorted(C); ax.bar(np.array(xs)+off,[C[x] for x in xs],0.36,color=c,label=f'{fam} (n={len(q)})')
ax.set_xlabel(r'per-core net charge $q=Q_{\rm hole}-N_-^{\rm local}$'); ax.set_ylabel('count'); ax.legend(fontsize=8)
ax.set_title('(c) charge absorbed by one core\nlens prediction: twist 2πθC/A_c≈0.97, mismatch 0',fontsize=10)
ax=fig.add_subplot(2,3,4)
Lm={0:0,1:3,2:5,3:6,4:7,5:8,6:9,7:9}
C=Counter((abs(o['w']),o['z']) for o in ZW)
for (w,z),n in C.items(): ax.scatter(w,z,s=12+6*n,color='#2471a3',alpha=.6)
ww=np.arange(0,8); ax.step(ww,[2*Lm[w] for w in ww],where='mid',c='#c0392b',label=r'bound $z\geq 2L_{\min}(|w|)$')
ax.set_xlabel('|w| (AA winding of circle around core)'); ax.set_ylabel('z (walls crossing the circle)'); ax.legend(fontsize=8)
ax.set_title('(d) wall valence vs winding: 444 circles,\n0 violations, z always even',fontsize=10)
ax=fig.add_subplot(2,3,5)
P=[r for r in R if r['fam']=='twist' and r['mode']=='pair']
for lam,mk in [(-0.5,'o'),(1.0,'s')]:
    for r in P:
        if r['lam']!=lam or r['branch'] not in ('pull','push'): continue
        a=r['Qbox']-r['Nplus']; ax.scatter(r['d'],a+(0.08 if lam>0 else -0.08),marker=mk,color=col[a],edgecolor='k',lw=.4,s=50)
ax.scatter([],[],marker='o',c='w',edgecolor='k',label='K/μ=0.5'); ax.scatter([],[],marker='s',c='w',edgecolor='k',label='K/μ=2')
ax.axhline(1.94,c='k',ls=':',lw=.8); ax.text(50,2.0,'ideal 1.94',fontsize=8)
ax.set_yticks([1,2,3]); ax.set_xlabel('d'); ax.set_ylabel('absorbed charge of the pair'); ax.legend(fontsize=8)
ax.set_title('(e) pull branch = charge-starved (1);\npush branch = 2 (or 3 at short d)',fontsize=10)
ax=fig.add_subplot(2,3,6); ax.axis('off')
txt=("Gauss law (checked exactly, 84 states):\n"
     "  N+ − N− + Σ_k Q_hole,k = Q_box\n\n"
     "single defect, twist:    Q_hole=5, N−=4  →  q=+1\n"
     "single defect, mismatch: Q_hole=2, N−=2  →  q= 0\n\n"
     "twist pair, d=40 (K/μ=0.5):\n"
     "  pull:  q=(+1, 0)  E=129.66\n"
     "  push:  q=(+1,+1)  E=125.41\n"
     "  NEB snapping = second core absorbs the\n"
     "  trapped AA node  (q: 0 → +1)\n\n"
     "mismatch pair, d=26:\n"
     "  q=(0,0)  E=161.06   q=(−1,0)  E=163.03\n"
     "  (the 'structural' branch that survived\n"
     "   the mirror constraint is a charge excitation)")
ax.text(0,1,txt,va='top',family='monospace',fontsize=9.2)
plt.tight_layout(); plt.savefig('topo_charges.png',dpi=120)
