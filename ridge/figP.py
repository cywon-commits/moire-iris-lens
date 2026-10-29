import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.ticker
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig=plt.figure(figsize=(15,7.6)); gs=fig.add_gridspec(2,4,hspace=0.42,wspace=0.42,height_ratios=[1,0.82])
def load(t):
    D=np.load(f'per_{t}.npz'); x=D['x']; d=x[1]-x[0]
    # tile 2x2 cells (periodic) for readability, centered
    T=lambda f: np.tile(f,(2,2))
    xx=np.concatenate([x,x+16]); return D,xx,T
dia=lambda ax: [ax.plot(*np.array(s).T,ls='--',color='#9be',lw=0.9) for s in
     [[(-8,0),(0,8)],[(0,8),(8,0)],[(8,0),(0,-8)],[(0,-8),(-8,0)],[(8,0),(16,8)],[(16,8),(24,0)],[(0,8),(8,16)],[(8,16),(16,8)],[(0,8),(-8,16)],[(8,0),(16,-8)],[(16,-8),(24,0)],[(0,-8),(8,-16)]]]
pan=[('P1','mis','(a) in-plane, no buckling\nwalls on |x|+|y|=L/2 (dashed)'),
     ('P4','h',r'(b) $h$: $d(u)$ only, $\Lambda=0.6$'),
     ('P2','h',r'(c) $h$: buckling only, $\Lambda=0.1$'),
     ('P5','h',r'(d) $h$: both, $\Lambda=0.1$, $d_1=0.3a$')]
for k,(t,kind,ttl) in enumerate(pan):
    ax=fig.add_subplot(gs[0,k]); D,xx,T=load(t); e=[xx[0],xx[-1],xx[0],xx[-1]]
    if kind=='mis':
        f=T(np.hypot(D['ux']-np.round(D['ux']),D['uy']-np.round(D['uy'])))
        im=ax.imshow(f.T,origin='lower',extent=e,cmap='magma',vmin=0,vmax=0.71); dia(ax); lab='|u mod a|'
    else:
        f=T(D['h']); m=0.62; im=ax.imshow(f.T,origin='lower',extent=e,cmap='RdBu_r',vmin=-m,vmax=m); lab='h / a'
    ax.set_xlim(xx[0],xx[-1]); ax.set_ylim(xx[0],xx[-1]); ax.set_title(ttl,fontsize=9.5)
    ax.set_xticks([-8,0,8,16,24]); ax.set_yticks([-8,0,8,16,24])
    cb=plt.colorbar(im,ax=ax,fraction=0.046,pad=0.03); cb.set_label(lab,fontsize=8); cb.ax.tick_params(labelsize=8)
# (e) lift ratio
Lam=np.array([0.6,1.0,3.0]); wall=np.array([0.258,0.190,0.150])/0.15; node=np.array([0.381,0.321,0.265])/0.30
ax=fig.add_subplot(gs[1,0:2])
ax.plot(Lam,wall,'o-',color='#2a6fb0',lw=2,ms=7,label=r'wall centre: $h/d$, $d=d_1/2$')
ax.plot(Lam,node,'s-',color='#c0392b',lw=2,ms=7,label=r'node: $h/d$, $d=d_1$')
ax.axhline(1,color='#888',lw=1,ls=':'); ax.axvline(0.509,color='#888',lw=1,ls='--')
ax.text(0.52,1.05,r'linear threshold $\Lambda_c$',fontsize=9,color='#555',rotation=90)
ax.set_xscale('log'); ax.set_xticks([0.5,0.6,1,2,3]); ax.set_xticklabels(['0.5','0.6','1','2','3']); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_xlabel(r'$\Lambda=\kappa K_z/YV_0$ ($\ell_b=0.5a$, $d_1=0.3a$)'); ax.set_ylabel('lift / geometric d(u)')
ax.set_title('(e) compression amplifies the stacking-driven lift',fontsize=10); ax.legend(frameon=False,fontsize=9)
# (f) cut across a wall along the diagonal x=y
ax=fig.add_subplot(gs[1,2:])
for t,c,lb in [('P7','#888','d only, Λ=3'),('P4','#2a6fb0','d only, Λ=0.6'),('P2','#c0392b','buckling only, Λ=0.1'),('P5','#6a3d9a','both, Λ=0.1')]:
    D=np.load(f'per_{t}.npz'); x=D['x']; N=len(x); i=np.arange(N); s=np.sqrt(2)*x
    ax.plot(s,D['h'][i,i],color=c,lw=2,label=lb)
ax.set_xlabel(r'distance along the diagonal $x=y$ (a); walls at $s=\pm 4\sqrt{2}$'); ax.set_ylabel('h / a')
[ax.axvline(v,color="#bbb",lw=0.8,ls="--") for v in (-4*np.sqrt(2),4*np.sqrt(2))]
ax.set_title('(f) cut through two walls',fontsize=10); ax.legend(frameon=False,fontsize=8.5,ncol=2)
fig.savefig('periodic_results.png',dpi=130,bbox_inches='tight')
