import json,glob,numpy as np,matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
J=lambda t: json.load(open(t+'.json'))['E']
R=np.sqrt(120.); ds=[8.,16.,26.,40.,56.]
# lambda=-0.5
L='_lam-0.5'; bg0=J('none_d0.0_V0.0_a0_x'+L)
def U(tag_suffix,bgtag):
    out=[]
    for d in ds:
        try:
            UV=J(f'pair_d{d}_V0.005_a0_x{tag_suffix}')-2*J(f'R_d{d}_V0.005_a0_x{tag_suffix}')+J(bgtag)
            U0=J(f'pair_d{d}_V0.0_a0_x'+L)-2*J(f'R_d{d}_V0.0_a0_x'+L)+bg0
            out.append((d,UV-U0))
        except FileNotFoundError: pass
    return np.array(out)
Uf=U(L,'none_d0.0_V0.005_a0_x'+L); Ua=U('_sym'+L,'none_d0.0_V0.005_a0_x_sym'+L)
# lambda=1
cs=json.load(open('chiral_sep.json'))
U1c=np.array([(r['d'],r['U_chiral']) for r in cs]); U1a=np.array([(r['d'],r['U_achiral']) for r in cs])
tw=np.array([(r[0],r[1]) for r in json.load(open('Uall.json'))['90_x']])
print('lam=-0.5 free',Uf.round(3).tolist()); print('lam=-0.5 achiral',Ua.round(3).tolist())
# chirality energies
def chi(lam):
    s='_x'+lam; r={}
    r['bg']=J('none_d0.0_V0.005_a0'+s.replace('_x','_x'))-J('none_d0.0_V0.005_a0_x_sym'+lam)
    for m in ['R','pair']:
        r[m]=[J(f'{m}_d{d}_V0.005_a0_x{lam}')-J(f'{m}_d{d}_V0.005_a0_x_sym{lam}') for d in ds if glob.glob(f'{m}_d{d}_V0.005_a0_x_sym{lam}.json')]
    return r
c1=chi(''); c5=chi(L); print('chi gain lam=1',c1); print('chi gain lam=-0.5',c5)
fig=plt.figure(figsize=(15,9.5))
ax=fig.add_subplot(2,3,(1,2))
ax.plot(U1c[:,0]/R,U1c[:,1],'o--',c='#c0392b',alpha=.6,label=r'$K/\mu=2$ free (lens + node-chirality Ising)')
ax.plot(U1a[:,0]/R,U1a[:,1],'s-',c='#c0392b',label=r'$K/\mu=2$ mirror-constrained (lens only)')
ax.plot(Uf[:,0]/R,Uf[:,1],'o--',c='#2471a3',alpha=.6,label=r'$K/\mu=0.5$ free')
ax.plot(Ua[:,0]/R,Ua[:,1],'s-',c='#2471a3',label=r'$K/\mu=0.5$ mirror-constrained (lens only)')
ax.plot(tw[:,0]/R,tw[:,1],'^:',c='gray',label=r'twist ($\alpha=90^\circ$, $K/\mu=2$) for reference')
ax.axhline(0,c='k',lw=.6); ax.axvline(1,c='k',ls=':',lw=.8); ax.axvline(2*np.sqrt(2),c='k',ls=':',lw=.8)
ax.text(1.03,ax.get_ylim()[1]*0.9 if False else 6.3,r'$d=R$',fontsize=9); ax.text(2.86,6.3,r'$d=2\sqrt{2}R$',fontsize=9)
ax.set_xlabel(r'$d/R$  ($R=\sqrt{C/\theta}\approx11$)'); ax.set_ylabel(r'$U(d)$ (moiré-mediated)'); ax.set_ylim(-1,7)
ax.set_title('(a) converging-lens pair: separating lens part from node-chirality part'); ax.legend(fontsize=8.5,loc='upper right')
ax=fig.add_subplot(2,3,3)
w=0.35; x5=np.arange(len(c5['pair']))
ax.bar(-0.9,c1['bg'],w,color='#c0392b'); ax.bar(-0.9+w,c5['bg'],w,color='#2471a3')
ax.bar(np.arange(len(c1['pair'])),c1['pair'],w,color='#c0392b',label=r'$K/\mu=2$')
ax.bar(x5+w,c5['pair'],w,color='#2471a3',label=r'$K/\mu=0.5$')
ax.set_xticks([-0.9+w/2]+list(np.arange(len(ds))+w/2)); ax.set_xticklabels(['bg']+[f'pair\nd={d:.0f}' for d in ds],fontsize=8)
ax.axhline(0,c='k',lw=.6); ax.set_ylabel(r'$E_{free}-E_{mirror}$'); ax.legend(fontsize=8)
ax.set_title('(b) energy gained by breaking mirror\n(chirality condensation)')
def wmap(ax,f,title,vm):
    D=np.load(f); x=D['x']; h=x[1]-x[0]; ux,uy=D['ux'],D['uy']
    wz=0.5*(np.gradient(uy,h,axis=0)-np.gradient(ux,h,axis=1)); wz[D['core']]=np.nan
    m=np.abs(x)<60; im=ax.imshow(wz[np.ix_(m,m)].T,origin='lower',extent=[x[m][0],x[m][-1]]*2,cmap='RdBu_r',vmin=-vm,vmax=vm)
    ax.set_title(title,fontsize=10); ax.set_xticks([]); ax.set_yticks([]); plt.colorbar(im,ax=ax,fraction=.046)
wmap(fig.add_subplot(2,3,4),'none_d0.0_V0.005_a0_x.npz',r'(c) $K/\mu=2$ background: local rotation $\omega$'+'\n(all nodes same hand, |ω|≈0.096)',0.15)
wmap(fig.add_subplot(2,3,5),'none_d0.0_V0.005_a0_x_lam-0.5.npz',r'(d) $K/\mu=0.5$ background'+'\n(|ω|≈0.01, mirror gain ≈ 0)',0.15)
wmap(fig.add_subplot(2,3,6),'pair_d26.0_V0.005_a0_x_lam-0.5.npz',r'(e) $K/\mu=0.5$, pair $d=26$'+'\n(lens-induced swirl near defects)',0.15)
plt.tight_layout(); plt.savefig('lens_vs_chirality.png',dpi=120)
