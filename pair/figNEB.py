import os
import json,numpy as np,matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, sys
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','hex')); from hexcore import Vhex
s3=np.sqrt(3); J=lambda t: json.load(open(t+'.json'))['E']
fig=plt.figure(figsize=(16,9.6))
ax=fig.add_subplot(2,3,1)
for tag,c,ls,lab in [('L05_d34c','#2471a3','-','K/μ=0.5, d=34'),('L2_d34c','#c0392b','-','K/μ=2, d=34'),('L05_d40b','#2471a3','--','K/μ=0.5, d=40'),('L2_d40b','#c0392b','--','K/μ=2, d=40')]:
    D=np.load(f'neb_{tag}.npz'); R=D['Rall']; s=np.r_[0,np.cumsum(np.linalg.norm(np.diff(R,axis=0),axis=1))]; E=D['E']
    ax.plot(s/s[-1],E-E[0],'o'+ls,c=c,ms=4,label=lab)
ax.axhline(0,c='k',lw=.6); ax.set_xlabel('reaction coordinate (normalized arc length)'); ax.set_ylabel('E − E(pulled)')
ax.set_title('(a) minimum-energy paths: pulled → pushed branch\n(NEB + climbing image)',fontsize=10); ax.legend(fontsize=8); ax.set_ylim(-5,1.6)
ax=fig.add_subplot(2,3,2)
b={'K/μ=0.5':[J('neb_L05_d34c') if False else None]}
def bar(tag): D=json.load(open(f'neb_{tag}.json')); E=np.array(D['E']); return E[1:6].max()-E[0] if True else 0
B=[[0.029,0.143],[0.923,0.258]]
x=np.arange(2); ax.bar(x,B[0],.35,color='#2471a3',label='K/μ=0.5'); ax.bar(x+.35,B[1],.35,color='#c0392b',label='K/μ=2')
for i in range(2):
    ax.text(x[i],B[0][i]+.02,f'{B[0][i]:.2f}',ha='center',fontsize=9); ax.text(x[i]+.35,B[1][i]+.02,f'{B[1][i]:.2f}',ha='center',fontsize=9)
ax.set_xticks(x+.17); ax.set_xticklabels(['d=34','d=40']); ax.set_ylabel('escape barrier from pulled branch (upper bound)')
ax.set_title('(b) escape barrier\n(still decreasing when stopped → upper bounds)',fontsize=10); ax.legend(fontsize=8)
ax=fig.add_subplot(2,3,3)
d5=[16,21,26,30,34,40,42,44,46]; e5=[J(f'swout_d{d:.1f}') for d in d5[:6]]+[J(f'swBout_d{d:.1f}') for d in d5[6:]]
d2=[16,21,26,30,34,40,48,52,56]; e2=[J(f'twL1out_d{d:.1f}') for d in d2[:7]]+[J(f'twL1Bout_d{d:.1f}') for d in d2[7:]]
ax.plot(d5,np.array(e5)-e5[0],'o-',c='#2471a3',label='K/μ=0.5'); ax.plot(d2,np.array(e2)-e2[0],'s-',c='#c0392b',label='K/μ=2')
ax.axvspan(40,42,color='#2471a3',alpha=.12); ax.axvspan(48,52,color='#c0392b',alpha=.12)
ax.set_xlabel('d'); ax.set_ylabel('E_pair(d) − E_pair(16) along pulling'); ax.legend(fontsize=8)
ax.set_title('(c) quasi-static pulling: release distance\nK/μ=0.5: d≈40–42 (partial rewiring);  K/μ=2: d≈48–52 (full snap)',fontsize=10)
def vmap(ax,R,N,x,core,t):
    ux,uy=R.reshape(2,N,N); V,_,_=Vhex(ux,uy,0.005,s3*0.005); V[core]=np.nan; m=np.abs(x)<42
    ax.imshow(V[np.ix_(m,m)].T,origin='lower',cmap='magma',extent=[x[m][0],x[m][-1]]*2); ax.set_title(t,fontsize=10); ax.set_xticks([]); ax.set_yticks([])
D=np.load('neb_L2_d34c.npz'); N=int(D['N']); x=D['x']; core=np.load('twL1out_d34.0.npz')['core']; R=D['Rall']; E=D['E']; i=int(D['imax'])
vmap(fig.add_subplot(2,3,4),R[0],N,x,core,'(d) K/μ=2, d=34, pulled state\nAA node trapped on the double wall between the cores')
vmap(fig.add_subplot(2,3,5),R[i],N,x,core,f'(e) saddle (image {i}), ΔE={E[i]-E[0]:+.2f}\nnode smears out along the double wall (AA region stretched)')
vmap(fig.add_subplot(2,3,6),R[-1],N,x,core,f'(f) pushed state, ΔE={E[-1]-E[0]:+.2f}\nnode re-wired into the network with vertical walls')
plt.tight_layout(); plt.savefig('neb_barrier.png',dpi=120)
