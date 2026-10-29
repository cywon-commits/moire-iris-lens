import numpy as np, json, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig=plt.figure(figsize=(13,9.2)); gs=fig.add_gridspec(2,4,hspace=0.38,wspace=0.45)
# (a) threshold
r=np.array([0.02,0.05,0.1,0.2,0.4,0.8,1.6,3.2,6.4]); Lc=np.array([0.00179,0.0097,0.0311,0.0820,0.1653,0.2626,0.3955,0.5808,0.7482])
a=fig.add_subplot(gs[0,0]); a.loglog(r,Lc,'o-',color='#2a6fb0',lw=2,ms=6,label='exact (linear)')
rr=np.logspace(-2,1,50); a.loglog(rr,np.pi**2*rr**2/2,'--',color='#888',lw=1.2,label=r'$\pi^2r^2/2$')
a.axhline(1,ls=':',color='#888',lw=1.2,label='1 (wide wall)')
a.plot([2.466,1.233],[0.5088,0.3381],'s',color='#c0392b',ms=8,mfc='none',mew=2,label='1D nonlinear')
a.set_ylim(1e-3,2); a.set_xlabel(r'$r=w_n/\ell_b$'); a.set_ylabel(r'$\Lambda_c=(\kappa K_z/YV_0)_c$')
a.set_title('(a) buckling threshold of a wall'); a.legend(fontsize=8,frameon=False)
# (b) f vs Lam
S=json.load(open('scan.json')); b=fig.add_subplot(gs[0,1])
for lb,c in [('0.5','#2a6fb0'),('1.0','#c0392b')]:
    A=np.array(S[lb]); b.plot(A[:,0],A[:,2],'o-',color=c,lw=2,ms=5,label=rf'$\ell_b={lb}a$')
b.set_xlabel(r'$\Lambda=\kappa K_z/YV_0$'); b.set_ylabel(r'absorbed fraction $f$'); b.set_title('(b) extra row taken by z'); b.legend(frameon=False,fontsize=9)
# (c) profiles
c=fig.add_subplot(gs[0,2:])
for lb,col in [('0.5','#2a6fb0'),('1.0','#c0392b')]:
    P=np.load(f'prof_lb{lb}.npz'); c.plot(P['y'],P['h'],color=col,lw=2,label=rf'$h(y)$, $\ell_b={lb}a$')
    c.plot(P['y'],P['v'],color=col,lw=1.2,ls='--',label=rf'$u_y(y)$, $\ell_b={lb}a$')
c.set_xlim(-8,8); c.set_xlabel('y / a  (across the +x half-axis wall)'); c.set_title(r'(c) wall profiles at $\Lambda\approx0.1$'); c.legend(frameon=False,fontsize=8,ncol=2)
# 2D
def mis(D): return np.hypot(D['ux']-np.round(D['ux']),D['uy']-np.round(D['uy']))
x=np.load('r2d_a.npz')['x']; e=[x[0],x[-1],x[0],x[-1]]
X,Yg=np.meshgrid(x,x,indexing='ij')
q1=(X>0)&(Yg>=0); q2=(X<=0)&(Yg>0); q3=(X<0)&(Yg<=0); q4=(X>=0)&(Yg<0)
panels=[('d','mis','(d) relaxed in-plane, no buckling\n'+r'$\Lambda=0.6$: pinwheel → 45° diamond'),
        ('b','h',r'(e) $h$, $\ell_b=0.5a$, $\Lambda=0.1$'),
        ('c','h',r'(f) $h$, $\ell_b=2a$, $\Lambda=0.1$'),
        ('e','h','(g) pure-twist pinwheel: stays,\n'+r'$h$, $\ell_b=0.5a$')]
for k,(t,kind,ttl) in enumerate(panels):
    ax=fig.add_subplot(gs[1,k]); D=np.load(f'r2d_{t}.npz')
    if kind=='mis': im=ax.imshow(mis(D).T,origin='lower',extent=e,cmap='magma',vmin=0,vmax=0.6)
    else:
        m=abs(D['h']).max(); im=ax.imshow(D['h'].T,origin='lower',extent=e,cmap='RdBu_r',vmin=-m,vmax=m)
        ax.contour(x,x,mis(D).T,levels=[0.25],colors='k',linewidths=0.5)
    ax.set_title(ttl,fontsize=9.5); ax.set_xticks([-10,0,10]); ax.set_yticks([-10,0,10])
    cb=plt.colorbar(im,ax=ax,fraction=0.046,pad=0.03); cb.ax.tick_params(labelsize=8)
    cb.set_label('|u mod a|' if kind=='mis' else 'h / a',fontsize=8)
fig.savefig('ridge_results.png',dpi=130,bbox_inches='tight')
