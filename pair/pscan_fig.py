"""P1 figure and summary table from pscan_results.json.
usage: python pscan_fig.py [pscan_results.json]  -> pscan.png, pscan_summary.md
Per (alpha,|C|,phi_m,placement) the lower-energy of the two initial conditions is the main result; placement averages are
over the four placements (AA, AB, SP, GEN). q_lin at AA with phi_m=0 is undefined (exact AA Einstein ring) and is excluded."""
import sys,json,numpy as np
from collections import defaultdict
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
R=[r for r in json.load(open(sys.argv[1] if len(sys.argv)>1 else 'pscan_results.json')) if r.get('status')=='converged']
main=[r for r in R if r.get('rhmode','s')=='s' and r['phim']!=-90]
best={}
for r in main:
    k=(r['alpha'],r['absC'],r['phim'],r['place'])
    if k not in best or r['dE']<best[k]['dE']: best[k]=r
G=defaultdict(list)
for k,r in best.items(): G[k[:3]].append(r)
cols={0:'C0',90:'C1',180:'C3'}; mk={0.:'o',90.:'s'}; lab={0.:'mismatch',90.:'twist'}
fig,ax=plt.subplots(1,3,figsize=(15,4.6))
rows=[]
for (a,c,ph),v in sorted(G.items()):
    q=np.array([r['q'] for r in v]); f=np.array([r['mdQ'] for r in v]); T=v[0]['T']
    ax[0].errorbar(T*(1-np.cos(np.radians(ph))),q.mean(),yerr=q.std(),fmt=mk[a],color=cols[int(ph)],mfc='none' if a==90 else cols[int(ph)],capsize=2)
    ax[1].errorbar(T*np.cos(np.radians(ph)),f.mean(),yerr=f.std(),fmt=mk[a],color=cols[int(ph)],mfc='none' if a==90 else cols[int(ph)],capsize=2)
    ql=[r['q_lin'] for r in v if not (r['place']=='AA' and ph==0)]
    rows.append((a,c,T,ph,len(v),q.mean(),q.std(),T*(1-np.cos(np.radians(ph))),np.mean(ql) if ql else np.nan,f.mean(),f.std(),T*np.cos(np.radians(ph)),
                 ' '.join(f"{r['place']}:{r['q']}" for r in sorted(v,key=lambda r:['AA','AB','SP','GEN'].index(r['place'])))))
for i,(xl,yl,t) in enumerate([('T(1-cos φ_m)','⟨q⟩ (placement mean ± sd)','(a) bound share'),('T cos φ_m','⟨−ΔQ⟩','(b) far share')]):
    lim=[-6,12] if i==0 else [-6,6]; ax[i].plot(lim,lim,'k--',lw=0.8); ax[i].set_xlabel(xl); ax[i].set_ylabel(yl); ax[i].set_title(t)
for ph in (0,90,180): ax[0].plot([],[],'o',color=cols[ph],label=f'φ_m={ph}°')
for a in (0.,90.): ax[0].plot([],[],mk[a],color='k',mfc='k' if a==0 else 'none',label=lab[a])
ax[0].legend(fontsize=8)
d=np.array([r['q']-r['q_lin'] for r in best.values() if not (r['place']=='AA' and r['phim']==0)])
ax[2].hist(d,bins=np.arange(d.min()-0.5,d.max()+1.5),color='0.5',edgecolor='k'); ax[2].set_xlabel('q − q_lin'); ax[2].set_ylabel('count')
ax[2].set_title(f'(c) relaxation vs rounding: q=q_lin in {np.mean(d==0)*100:.0f}% of {len(d)}')
plt.tight_layout(); plt.savefig('pscan.png',dpi=110)
L=['| 배경 | \\|C\\| | T | φ_m | n | ⟨q⟩ ± sd | T(1−cos φ_m) | ⟨q_lin⟩ | ⟨−ΔQ⟩ ± sd | T cos φ_m | 배치별 q |','|---|---|---|---|---|---|---|---|---|---|---|']
for a,c,T,ph,n,qm,qs,qp,ql,fm,fs,fp,s in rows:
    L.append(f"| {'mismatch' if a==0 else 'twist'} | {c:g} | {T:.2f} | {ph:.0f} | {n} | {qm:.2f} ± {qs:.2f} | {qp:.2f} | {ql:.2f} | {fm:+.2f} ± {fs:.2f} | {fp:+.2f} | {s} |")
open('pscan_summary.md','w').write('\n'.join(L)+'\n'); print('\n'.join(L))
ab=defaultdict(dict)
for r in main: ab[r['tag'][:-2]][r['init']]=r
pairs=[v for v in ab.values() if 'a' in v and 'b' in v]
print('init pairs',len(pairs),'q differs',sum(v['a']['q']!=v['b']['q'] for v in pairs),'lower-E init b',sum(v['b']['dE']<v['a']['dE'] for v in pairs))
print('gauss!=0',sum(r['gauss']!=0 for r in R),' q==q_lin',np.mean(d==0))
