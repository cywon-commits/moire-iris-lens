import json,numpy as np,matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
R=json.load(open('conserv_res.json'))
fig,axs=plt.subplots(1,3,figsize=(18,5.4))
ax=axs[0]
for k,c in [('mm def (C=4)','#2471a3'),('mm add (C=-4)','#c0392b'),('tw def','#7f8c8d'),('tw torque -','#8e44ad')]:
    v=R[k]; r=np.array(v['r']); ax.plot(r,v['dQ'],'-',c=c,lw=.8,alpha=.6); ax.plot(r,v['dQlin'],'-',c=c,lw=2.5,label=k)
ax.axvspan(36,58,color='k',alpha=.05); ax.axhline(0,c='k',lw=.5)
ax.set_xlabel('loop radius ρ around the defect'); ax.set_ylabel('ΔQ(ρ) = Q(ρ) − Q_bg(ρ)')
ax.set_title('(a) single defects: integer loop charge (thin)\nvs linear prediction from ΣC (thick)',fontsize=10); ax.legend(fontsize=8)
ax=axs[1]; P=[];M=[]
for k,v in R.items():
    r=np.array(v['r']); m=(r>=36)&(r<58); p=np.mean(np.array(v['dQlin'])[m]); q=np.mean(np.array(v['dQ'])[m]); P.append(p); M.append(q)
    mk='o' if v['fam']=='single' else 's'; col='#2471a3' if k.startswith('mm') else '#c0392b'
    ax.scatter(p,q,marker=mk,s=55,color=col,edgecolor='k',lw=.4,zorder=3)
P=np.array(P);M=np.array(M); sl=np.sum(P*M)/np.sum(P*P); cc=np.corrcoef(P,M)[0,1]
xx=np.linspace(-2.2,1.3,10); ax.plot(xx,xx,'k--',lw=1,label='measured = predicted')
ax.scatter([],[],marker='o',color='w',edgecolor='k',label='single defects'); ax.scatter([],[],marker='s',color='w',edgecolor='k',label='pairs (all branches)')
ax.scatter([],[],color='#2471a3',label='mismatch'); ax.scatter([],[],color='#c0392b',label='twist')
ax.set_xlabel('linear prediction  −(2π/A_c)·Re(κ̄ ΣC)  (+ box images)'); ax.set_ylabel('measured ⟨ΔQ⟩ over ρ∈[36,58]')
ax.set_title(f'(b) far node charge is fixed by ΣC\nslope {sl:.2f}, correlation {cc:.2f} (17 states)',fontsize=10); ax.legend(fontsize=8)
ax=axs[2]
for k,c in [('tw ++ d40 pull','#d35400'),('tw ++ d40 push','#1e8449'),('mm ++ d26 pull','#5dade2'),('mm ++ d26 push','#1b4f72')]:
    v=R[k]; ax.plot(v['r'],v['dQ'],'-',c=c,label=k,lw=1.3)
ax.plot(R['tw ++ d40 pull']['r'],R['tw ++ d40 pull']['dQlin'],'k:',lw=1.5,label='linear, twist ++')
ax.plot(R['mm ++ d26 pull']['r'],R['mm ++ d26 pull']['dQlin'],'k--',lw=1.5,label='linear, mismatch ++')
ax.axhline(0,c='k',lw=.5); ax.set_xlabel('ρ (around pair centre)'); ax.set_ylabel('ΔQ(ρ)')
ax.set_title('(c) different integer branches: same ΣC, same box total,\nbut charge-neutral rearrangements reach ρ≈50',fontsize=10); ax.legend(fontsize=7.5)
plt.tight_layout(); plt.savefig('conservation_test.png',dpi=115)
print('slope',sl,'corr',cc)
