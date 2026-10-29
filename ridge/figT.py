import numpy as np, json, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig=plt.figure(figsize=(15,8.4)); gs=fig.add_gridspec(2,4,hspace=0.45,wspace=0.42)
H=json.load(open('scan3_hel.json')); A_=json.load(open('scan3_alt.json'))
ax=fig.add_subplot(gs[0,0:2])
for D,c,lab,side in [(H,'#c0392b','helical (θ, θ)',14.335),(A_,'#2a6fb0','alternating (θ, −θ)',11.806)]:
    D=np.array(D); ax.plot(D[:,0],D[:,1]-D[-1,1],'o-',color=c,lw=2,ms=6,label=lab)
    ax.plot([0.5],[side-D[-1,1]],'D',color=c,ms=8,mfc='none',mew=1.8)
ax.axhline(0,color='#999',lw=0.8,ls=':')
ax.set_xlabel(r'offset of node lattice 23 along the diagonal, $\delta=t\,(L/2)(1,1)$')
ax.set_ylabel(r'$E(t)-E(1)$  (per moiré cell)')
ax.set_title('(a) node–node interaction across the middle layer\n(open diamonds: offset (L/2, 0), node on a wall)',fontsize=10)
ax.legend(frameon=False)
def mis(u): return np.hypot(*(u-np.round(u)))
pan=[('tri_hel_free.npz','(b) helical, free relaxation\nnodes staggered by (L/2, L/2)'),('tri_alt_0.npz','(c) alternating, coincident\nnodes stacked: 8 walls in projection')]
for k,(f,ttl) in enumerate(pan):
    D=np.load(f); x=D['x']; U=D['U']; d=x[1]-x[0]
    m12=np.tile(mis(U[1]-U[0]),(2,2)); m23=np.tile(mis(U[2]-U[1]),(2,2)); xx=np.concatenate([x,x+16])
    rgb=np.zeros(m12.shape+(3,)); rgb[...,0]=m12/0.71; rgb[...,2]=m23/0.71; rgb[...,1]=0.35*np.minimum(m12,m23)/0.71
    ax=fig.add_subplot(gs[0,2+k]); ax.imshow(np.transpose(rgb,(1,0,2)).clip(0,1),origin='lower',extent=[xx[0],xx[-1]+d,xx[0],xx[-1]+d])
    ax.set_title(ttl,fontsize=10); ax.set_xticks([-8,0,8,16,24]); ax.set_yticks([-8,0,8,16,24])
    ax.text(0.02,-0.2,'red: interface 1–2   blue: interface 2–3',transform=ax.transAxes,fontsize=8.5,color='#555')
# sigma_z
r=[]
for f in ['sigz_0.5_0.05,0.15,0.3,0.6.json','sigz_0.5_0.1,0.2,0.4,1.0.json']: r+=json.load(open(f))
r=np.array(sorted(r))
ax=fig.add_subplot(gs[1,0:2])
ax.plot(r[:,0],r[:,1],'o-',color='#6a3d9a',lw=2,ms=6,label=r'middle layer $\max|h_2|$ (σ$_z$-odd)')
ax.plot(r[:,0],r[:,3],'s-',color='#1d9e75',lw=2,ms=6,label=r'breathing $\max|h_3-h_1|$ (σ$_z$-even)')
ax.axvline(0.41,color='#888',ls='--',lw=1); ax.text(0.43,0.55,r'$\Lambda_c\approx0.41$'+'\n(bilayer 0.51)',fontsize=9,color='#555')
ax.set_xlabel(r'$\Lambda=\kappa K_z/YV_0$  ($\ell_b=0.5a$, $d_1=0.3a$, encapsulated)'); ax.set_ylabel('amplitude / a')
ax.set_title(r'(d) $d(u)$ drives only the σ$_z$-even breathing; $h_2$ is a pitchfork',fontsize=10); ax.legend(frameon=False,fontsize=9)
for k,(Lam,ttl) in enumerate([(1.0,r'(e) $h_2$ at $\Lambda=1$ (symmetric)'),(0.1,r'(f) $h_2$ at $\Lambda=0.1$ (σ$_z$ broken)')]):
    D=np.load(f'sigz_0.5_{Lam}.npz'); x=D['x']; h2=np.tile(D['H'][1],(2,2)); xx=np.concatenate([x,x+16])
    ax=fig.add_subplot(gs[1,2+k]); m=0.6
    im=ax.imshow(h2.T,origin='lower',extent=[xx[0],xx[-1],xx[0],xx[-1]],cmap='RdBu_r',vmin=-m,vmax=m)
    ax.set_title(ttl,fontsize=10); ax.set_xticks([-8,0,8,16,24]); ax.set_yticks([-8,0,8,16,24])
    cb=plt.colorbar(im,ax=ax,fraction=0.046,pad=0.03); cb.set_label(r'$h_2/a$',fontsize=8); cb.ax.tick_params(labelsize=8)
fig.savefig('trilayer_results.png',dpi=130,bbox_inches='tight')
