import json,numpy as np,matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
O=json.load(open('opp_results.json')); Lc=json.load(open('lenscharge.json')); R=np.sqrt(120.)
d=np.array([o['d'] for o in O])/R
fig,axs=plt.subplots(1,4,figsize=(19,4.6))
ax=axs[0]; ax.plot(d,[o['U0pm'] for o in O],'o-',c='#8e44ad',label='(+,−) closing + opening'); ax.plot(d,[o['U0pp'] for o in O],'s--',c='#555',label='(+,+) two closing')
ax.axhline(0,c='k',lw=.6); ax.set_xlabel('d/R'); ax.set_ylabel('U (V0=0, pure elasticity)'); ax.legend(fontsize=8)
ax.set_title('(a) elasticity alone: sign-blind\n(Eshelby: no long-range force)',fontsize=10)
ax=axs[1]; pm=np.array(Lc['pm']); pp=np.array(Lc['pp'])
ax.plot(pm[:,0]/R,pm[:,1],'o-',c='#8e44ad',label='(+,−)'); ax.plot(pp[:,0]/R,pp[:,1],'s--',c='#555',label='(+,+)')
ax.axhline(2*Lc['single'],c='k',ls=':',lw=.8); ax.text(4,1.98,'2 × 0.97',fontsize=8)
ax.set_xlabel('d/R'); ax.set_ylabel('ordinary node charge removed'); ax.legend(fontsize=8)
ax.set_title(r'(b) lens: removed charge $\propto\!\int\min(1,|f^\prime|^2)$'+'\nsubadditive for (+,−): |C₁+C₂| < |C₁|+|C₂|',fontsize=10)
for k,(a,t) in enumerate([(0,'(c) mismatch α=0, K/μ=0.5'),(90,'(d) twist α=90°, K/μ=0.5')]):
    ax=axs[2+k]; ax.plot(d,[o[f'Upm{a}'] for o in O],'o-',c='#8e44ad',label='(+,−)'); ax.plot(d,[o[f'Upp{a}'] for o in O],'s--',c='#555',label='(+,+) default runs')
    for o in O: ax.annotate(f"{o[f'abs_pm{a}']}",(o['d']/R,o[f'Upm{a}']),xytext=(4,-12),textcoords='offset points',fontsize=8,color='#8e44ad')
    ax.axhline(0,c='k',lw=.6); ax.set_xlabel('d/R'); ax.set_ylabel('moiré-mediated U'); ax.legend(fontsize=8)
    ax.set_title(t+'\n(numbers: absorbed node charge of the (+,−) pair)',fontsize=10)
plt.tight_layout(); plt.savefig('opp_pair.png',dpi=115)
