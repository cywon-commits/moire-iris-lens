import numpy as np, json, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from quartic import count_grid
from caustic import critical
plt.rcParams.update({'font.size':10})
cm=ListedColormap(['#f2f2f2','#cfe3f5','#f5c4b3']); nm=BoundaryNorm([-0.5,1,3,5],3)   # 0 / 2 / 4 images
fig=plt.figure(figsize=(19,13)); gs=fig.add_gridspec(3,4,hspace=0.3,wspace=0.25)
for r,(name,phd) in enumerate([('converging  φ_m = 0°',0),('twist (vacancy)  φ_m = −90°',-90),('diverging  φ_m = 180°',180)]):
    m=np.exp(1j*np.radians(phd))
    for c,a in enumerate([0.0,0.35,0.8]):
        b,N,D=count_grid(a,m,L=3.0,n=201)
        ax=fig.add_subplot(gs[r,c]); ax.imshow(N.T,origin='lower',extent=[b[0],b[-1],b[0],b[-1]],cmap=cm,norm=nm,interpolation='nearest')
        if a>0 or phd!=0:
            zc,bc=critical(a,m); ax.plot(bc.real,bc.imag,'k-',lw=1.2)
        else: ax.plot(0,0,'k.',ms=6)
        ax.add_patch(plt.Circle((0,0),1,fill=False,ec='#888',ls=':',lw=0.8))
        ax.set_xlim(-3,3); ax.set_ylim(-3,3); ax.set_aspect('equal'); ax.set_xticks([-2,0,2]); ax.set_yticks([-2,0,2])
        if c==0: ax.set_ylabel(name+'\nβ_y / R')
        if r==2: ax.set_xlabel('β_x / R   (original node position)')
        ax.set_title(f'shear |a| = {a}',fontsize=10)
ph=json.load(open('phase.json')); P=sorted(int(k) for k in ph); A=[0.1,0.2,0.35,0.5,0.7,0.9]
ax=fig.add_subplot(gs[0:2,3])
A4=np.array([[ph[str(p)][i]['A4'] for i in range(6)] for p in P]); A0=np.array([[ph[str(p)][i]['A0'] for i in range(6)] for p in P])
pe=np.array([-7.5]+[p+7.5 for p in P]); ae=np.array([0.05,0.15,0.275,0.425,0.6,0.8,1.0])
im=ax.pcolormesh(pe,ae,np.log10(A4.T+1e-3),cmap='Oranges',vmin=-2,vmax=1,shading='flat'); ax.set_ylim(0.05,1.0)
th=[(15,0.132),(30,0.216),(60,0.434),(90,0.581),(120,0.711),(150,0.817)]
ax.plot([t[0] for t in th],[t[1] for t in th],'ko-',ms=5,label='threshold for 4 images (numerical)')
pp=np.linspace(0,180,100); ax.plot(pp,np.sin(np.radians(pp)/2),'k:',lw=1,label='sin(φ_m/2)')
ax.set_yticks([0.1,0.2,0.35,0.5,0.7,0.9]); ax.set_xlabel('mass phase φ_m (deg)'); ax.set_ylabel('relative shear |a| = |λ/κ|')
ax.set_title('Einstein-cross (4-image) region: area (log, colour)\nopens for any shear in a converging lens,\nneeds |a| ≳ 0.8 sin(φ_m/2) otherwise',fontsize=10)
ax.legend(frameon=False,fontsize=8.5,loc='upper left'); plt.colorbar(im,ax=ax,fraction=0.04,label='log10 area of 4-image region (R²)')
ax=fig.add_subplot(gs[2,3]); ax.axis('off')
ax.text(0,0.95,"white: 0 images (node swallowed)\nblue:  2 images (1 normal + 1 mirrored)\norange: 4 images (2 normal + 2 mirrored)\nblack line: caustic (fold), cusps at its corners\n\ncount rule (singular core):\n  n_normal − n_mirrored = 0 everywhere\n\nwithout shear the converging caustic\nis a single point → Einstein ring",va='top',fontsize=10.5,family='monospace')
fig.savefig('caustics.png',dpi=95,bbox_inches='tight')
