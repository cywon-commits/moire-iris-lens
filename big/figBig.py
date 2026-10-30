import json,numpy as np,matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, sys, os
R=json.load(open('big_res.json')); C=json.load(open('ceff_compare.json')); F=json.load(open('ceff_fit.json'))
th=1/30.; Ac=np.sqrt(3)/2; k=0.2418
names=[n for n in ['tw pull d40','tw push d40','mm pull d26','mm push d26'] if n in R and n in C]
col={'tw pull d40':'#d35400','tw push d40':'#1e8449','mm pull d26':'#5dade2','mm push d26':'#1b4f72'}
fig,axs=plt.subplots(1,3,figsize=(18,5.2))
ax=axs[0]
for n in names:
    r=np.array(R[n]['r']); ax.plot(r,R[n]['dQ'],'-',c=col[n],lw=.6,alpha=.25); ww=np.ones(61)/61; ax.plot(r[30:-30],np.convolve(R[n]['dQ'],ww,'valid'),'-',c=col[n],lw=1.2,marker='o',ms=2.5,markevery=6,label=f'{n}: measured (running mean over 30)')
    p=np.array(C[n]['pred']); w=np.ones(21)/21; ax.plot(r,np.convolve(p,w,'same'),'-',c=col[n],lw=2.5,label=f'{n}: dressed-charge prediction')
    ax.plot(r,R[n]['dQlin'],':',c=col[n],lw=1.5)
ax.set_ylim(-4,4); ax.axhline(0,c='k',lw=.5); ax.set_xlim(20,140); ax.set_xlabel('loop radius ρ (box half-width 148.5)'); ax.set_ylabel('ΔQ(ρ)')
ax.set_title('(a) measured loop charge (dots: running mean) vs −(2π/A_c)Re(κ̄ C_eff) (thick);\ndotted = bare-C prediction',fontsize=10); ax.legend(fontsize=8)
ax=axs[1]
for n in names:
    r=np.array(C[n]['r']); w=np.ones(21)/21
    ax.plot(r,np.convolve(C[n]['ReC'],w,'same'),'-',c=col[n],lw=2,label=f'{n}: Re C_eff (area)')
    ax.plot(r,np.convolve(C[n]['ImC'],w,'same'),'--',c=col[n],lw=2,label=f'{n}: Im C_eff (torque)')
ax.axhline(8,c='k',ls=':',lw=1); ax.text(22,8.3,'bare ΣC = 8 (area), 0 (torque)',fontsize=8.5)
ax.axhline(0,c='k',lw=.5); ax.set_xlim(20,140); ax.set_xlabel('ρ'); ax.set_ylabel('C_eff(ρ) = −(1/2π)∮δu·(n + i t)')
ax.set_title('(b) dressed charge: the moiré adds a branch-dependent\narea or torque monopole (screened by the clamped edge ∝ 1 − ρ²/L²)',fontsize=10); ax.legend(fontsize=7.5)
ax=axs[2]
rows=[]
for n in names:
    r=np.array(C[n]['r']); m=(r>=50)&(r<80)
    rows.append((n,np.mean(np.array(C[n]['ReC'])[m]),np.mean(np.array(C[n]['ImC'])[m]),R[n]['q'],R[n]['E']))
y=np.arange(len(rows))
ax.barh(y-0.2,[r_[1] for r_ in rows],0.4,color='#7f8c8d',label='Re C_eff (area)')
ax.barh(y+0.2,[r_[2] for r_ in rows],0.4,color='#8e44ad',label='Im C_eff (torque)')
ax.axvline(8*0.8,c='k',ls=':',lw=1); ax.text(8*0.8+0.1,len(rows)-0.6,'bare ΣC, screened\nto ρ≈65 (≈6.4)',fontsize=8)
ax.set_yticks(y); ax.set_yticklabels([f'{r_[0]}\nq={r_[3]}, E={r_[4]:.1f}' for r_ in rows],fontsize=8.5); ax.axvline(0,c='k',lw=.6)
ax.set_xlabel('dressed charge, mean over ρ ∈ [50, 80]'); ax.legend(fontsize=8,loc='lower left')
ax.set_title('(c) same ΣC, different integer branch → different dressed charge\n(twist: torque part, couples via κ = iθ; mismatch: area part)',fontsize=10)
plt.tight_layout(); plt.savefig('big_box_test.png',dpi=115)
