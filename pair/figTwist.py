import json,numpy as np,matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
J=lambda t: json.load(open(t+'.json'))['E']; R=np.sqrt(120.); ds=[8.,16.,26.,40.,56.]; L='_lam-0.5'
U0={d:J(f'pair_d{d}_V0.0_a0_x'+L)-2*J(f'R_d{d}_V0.0_a0_x'+L)+J('none_d0.0_V0.0_a0_x'+L) for d in ds}
def U(sfx,bg): return np.array([(d,J(f'pair_d{d}_V0.005{sfx}')-2*J(f'R_d{d}_V0.005{sfx}')+J(bg)-U0[d]) for d in ds])
tw5=U(L,'none_d0.0_V0.005'+L); mm5=U('_a0_x_sym'+L,'none_d0.0_V0.005_a0_x_sym'+L)
tw1=np.array([(r[0],r[1]) for r in json.load(open('Uall.json'))['90_x']])
mm1=np.array([(r['d'],r['U_achiral']) for r in json.load(open('chiral_sep.json'))])
print('twist K/mu=0.5',tw5.round(3).tolist())
fig=plt.figure(figsize=(15,9.2))
ax=fig.add_subplot(2,3,(1,2))
ax.plot(tw1[:,0]/R,tw1[:,1],'^:',c='#7d3c98',alpha=.7,label=r'twist $\alpha=90^\circ$, $K/\mu=2$')
ax.plot(tw5[:,0]/R,tw5[:,1],'^-',c='#7d3c98',lw=2,label=r'twist $\alpha=90^\circ$, $K/\mu=0.5$')
ax.plot(mm1[:,0]/R,mm1[:,1],'s:',c='#c0392b',alpha=.7,label=r'mismatch $\alpha=0$, $K/\mu=2$ (lens only)')
ax.plot(mm5[:,0]/R,mm5[:,1],'s-',c='#2471a3',lw=2,label=r'mismatch $\alpha=0$, $K/\mu=0.5$ (lens only)')
for xv,t in [(1,r'$d=R$'),(np.sqrt(2),r'$\sqrt{2}R$'),(2*np.sqrt(2),r'$2\sqrt{2}R$')]:
    ax.axvline(xv,c='k',ls=':',lw=.8); ax.text(xv+.03,2.8,t,fontsize=9)
ax.axhline(0,c='k',lw=.6); ax.set_ylim(-1.2,3.2); ax.set_xlabel(r'$d/R$'); ax.set_ylabel(r'$U(d)$')
ax.set_title(r'(a) twist: $U(d)$ independent of $K$ (shear sector)  vs  mismatch: $U\propto$ roughly $K$ (dilation sector)',fontsize=10.5)
ax.legend(fontsize=8.5,loc='upper right')
ax=fig.add_subplot(2,3,3)
lab=['twist bg','mismatch bg\n(sym)']
E1=[J('none_d0.0_V0.005'),J('none_d0.0_V0.005_a0_x_sym')-J('none_d0.0_V0.0_a0_x')]
E5=[J('none_d0.0_V0.005'+L),J('none_d0.0_V0.005_a0_x_sym'+L)-J('none_d0.0_V0.0_a0_x'+L)]
xx=np.arange(2); ax.bar(xx,E1,.35,color='#c0392b',label=r'$K/\mu=2$'); ax.bar(xx+.35,E5,.35,color='#2471a3',label=r'$K/\mu=0.5$')
for i in range(2): ax.text(xx[i]+.17,max(E1[i],E5[i])+3,f'{E1[i]:.1f} / {E5[i]:.1f}',ha='center',fontsize=9)
ax.set_xticks(xx+.17); ax.set_xticklabels(lab); ax.set_ylabel('relaxed moiré energy (minus affine elastic)')
ax.set_title('(b) background moiré energy\n(twist: K-blind, mismatch: K-sensitive)',fontsize=10); ax.legend(fontsize=8); ax.set_ylim(0,150)
def wmap(ax,f,title,d):
    D=np.load(f); x=D['x']; h=x[1]-x[0]; ux,uy=D['ux'],D['uy']
    wz=0.5*(np.gradient(uy,h,axis=0)-np.gradient(ux,h,axis=1)); wz[D['core']]=np.nan
    m=np.abs(x)<60; im=ax.imshow(wz[np.ix_(m,m)].T,origin='lower',extent=[x[m][0],x[m][-1]]*2,cmap='RdBu_r',vmin=-.3,vmax=.3)
    for s in (-1,1):
        ax.add_patch(plt.Circle((s*d/2,0),np.sqrt(2)*R,fill=False,ls='--',ec='k',lw=.8))
    ax.set_title(title,fontsize=10); ax.set_xticks([]); ax.set_yticks([]); return im
for i,d in enumerate([16.,26.,40.]):
    im=wmap(fig.add_subplot(2,3,4+i),f'pair_d{d}_V0.005_lam-0.5.npz',rf'({"cde"[i]}) twist pair $d={d:.0f}$ ($d/R={d/R:.2f}$), $U={tw5[[8,16,26,40,56].index(int(d)),1]:+.2f}$'+'\nlocal rotation ω; dashed: swallow radius $\\sqrt{2}R$',d)
plt.colorbar(im,ax=fig.axes[-1],fraction=.046)
plt.tight_layout(); plt.savefig('twist_pair.png',dpi=120)
