import numpy as np, json, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.ticker
from hexcore import Vhex
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig=plt.figure(figsize=(17,8.6)); gs=fig.add_gridspec(2,4,hspace=0.32,wspace=0.28)
def vmap(ax,f,ttl,circ=None):
    D=np.load(f); x=D['x']; V=Vhex(D['ux'],D['uy'],1.,np.sqrt(3))[0]
    ax.imshow(V.T,origin='lower',extent=[x[0],x[-1],x[0],x[-1]],cmap='magma',vmin=0,vmax=9)
    if circ: ax.add_patch(plt.Circle((0,0),circ,fill=False,ec='#5ad',lw=1.5,ls='--'))
    ax.set_title(ttl,fontsize=10); ax.set_xticks([-16,0,16]); ax.set_yticks([-16,0,16])
vmap(fig.add_subplot(gs[0,0]),'ten_lam_1.0.npz','(a1) start: 10 walls at origin\nrelaxed, λ/μ=1')
vmap(fig.add_subplot(gs[0,1]),'ten_lam_15.0.npz','(a2) λ/μ=15: still two nodes,\nconnecting wall tensile')
vmap(fig.add_subplot(gs[0,2]),'ten_lam_40.0.npz','(a3) λ/μ=40: connecting wall\nrotates toward shear')
ax=fig.add_subplot(gs[0,3])
rows=[(1.0,18.0)]+[(r['v'],r['D']) for r in json.load(open('ten_lam_3,6,10,15,25,40.json'))]
rows=np.array(rows); ax.plot(rows[:,0]+2,rows[:,1],'o-',color='#2a6fb0',lw=2,ms=6)
ax.axvline(10.5,color='#c0392b',ls='--',lw=1); ax.text(11,5,'naive threshold\n√(Y/μ)=3.24',fontsize=8.5,color='#c0392b')
ax.set_xscale('log'); ax.set_xticks([3,5,10,20,40]); ax.set_xticklabels(['3','5','10','20','40']); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); ax.set_ylim(0,22); ax.set_xlabel('Y/μ = (λ+2μ)/μ'); ax.set_ylabel('node separation D (a)'); ax.set_title('(a4) incompressibility does not merge',fontsize=10)
vmap(fig.add_subplot(gs[1,0]),'ten_pin_1.5.npz','(b1) AA-favouring defect p=1.5, σ=1.5a',circ=1.5)
vmap(fig.add_subplot(gs[1,1]),'ten_pin_6.0.npz','(b2) p=6, σ=1.5a: one node captured',circ=1.5)
vmap(fig.add_subplot(gs[1,2]),'tensig_5.0_4.0.npz','(b3) p=4, σ=5a: still only one',circ=5.0)
ax=fig.add_subplot(gs[1,3])
pin=[(0.0,18.0)]+[(r['v'],r['D']) for r in json.load(open('ten_pin_0.5,1,1.5,2,3,4,6.json'))]
pin=np.array(pin); ax.plot(pin[:,0],pin[:,1],'o-',color='#6a3d9a',lw=2,ms=6,label='σ=1.5a')
sg=json.load(open('tensig.json'))
for s,c in [(2.5,'#b07cc6'),(3.5,'#e67e22'),(5.0,'#c0392b')]:
    pts=np.array([(r['p'],r['D']) for r in sg if r['sig']==s]); ax.plot(pts[:,0],pts[:,1],'s',color=c,ms=7,label=f'σ={s}a')
ax.set_ylim(0,22); ax.set_xlabel('defect strength p  (V → V·(1−p·g))'); ax.set_ylabel('node separation D (a)')
ax.set_title('(b4) pinning captures exactly one node',fontsize=10); ax.legend(frameon=False,fontsize=8.5)
fig.savefig('ten_results.png',dpi=110,bbox_inches='tight')
